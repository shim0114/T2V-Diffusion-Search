# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.my_fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from my_reward_model import VideoRewardCalculator

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import (usp_attn_forward, usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 # DLBS params 
                 num_beams: int = 1,
                 num_candidates: int = 1,
                 num_backtrack_steps: int = 1,
                 reward_model: VideoRewardCalculator = None, 
                 logging_file: str = None,):

        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2]
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.patch_size[1] * self.patch_size[2]) *
            target_shape[1] / self.sp_size
        ) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # --- Text Encoder ---
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        candidates_scores = [0.0 for _ in range(num_beams)]
        candidates_latents = []

        for _ in range(num_beams):
            noise = [
                torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=seed_g
                )
            ]
            candidates_latents.append(noise)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # --- Diffusion Sampling ---
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False
                )
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                    algorithm_type="sde-dpmsolver++",
                    final_sigmas_type="zero"
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas
                )
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for i, t in enumerate(tqdm(timesteps)):

                # --- Logging only on rank0 ---
                if self.rank == 0:
                    with open(logging_file, 'a') as f:
                        f.write(f'timestep: {t}\n')

                next_candidates_latents = []
                next_candidates_scores = []

                for b_idx, current_beam_latents in enumerate(candidates_latents):

                    latent_model_input = current_beam_latents
                    timestep = torch.stack([t])  # shape [1], same device

                    # -- model forward --
                    self.model.to(self.device)
                    noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    candidate_prev_latents = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        current_beam_latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                        num_candidates=num_candidates
                    )

                    # --- Generate for each candidate ---
                    for k_idx in range(num_candidates):

                        if t != timesteps[-1]:
                            next_timestep = timesteps[i+1]
                            latent_model_input_bt = [candidate_prev_latents[k_idx].squeeze(0).clone()]
                            
                            for bts in range(num_backtrack_steps, 0, -1):
                                t_bt = (next_timestep * bts) / num_backtrack_steps
                                t_next_bt = (next_timestep * (bts-1)) / num_backtrack_steps
                                
                                timestep_bt = torch.stack([t_bt])

                                self.model.to(self.device)
                                noise_pred_cond = self.model(latent_model_input_bt, t=timestep_bt, **arg_c)[0]
                                noise_pred_uncond = self.model(latent_model_input_bt, t=timestep_bt, **arg_null)[0]
                                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                                tmp_x0 = sample_scheduler.my_step(
                                    noise_pred.unsqueeze(0),
                                    t_bt,
                                    t_next_bt,
                                    latent_model_input_bt[0]
                                )
                                latent_model_input_bt = [tmp_x0.squeeze(0).clone()]
                                
                            x0 = latent_model_input_bt 

                            if offload_model:
                                self.model.cpu()
                                torch.cuda.empty_cache()

                            # ====== Modification: Only rank0 calculates Reward & logs ======
                            if self.rank == 0:
                                videos = self.vae.decode(x0)
                                new_score, score_details = reward_model(
                                    (
                                        (
                                            videos[0].unsqueeze(0)
                                            .permute(0, 2, 1, 3, 4)[:, :32:2, :, :, :]
                                            .squeeze(0) + 1
                                        ) / 2 * 255
                                    ).to(dtype=torch.uint8),
                                    input_prompt
                                )

                                # Logging only on rank0
                                with open(logging_file, 'a') as f:
                                    f.write(f'score_details: {score_details}\n')
                            else:
                                # Use dummy value on non-rank0 (can also use Broadcast for synchronization later)
                                new_score = 0.0

                            # --- Ensure all ranks have the same number of scores ---
                            # Here, we broadcast the "new_score calculated on rank0" to other ranks
                            new_score_t = torch.tensor([new_score], dtype=torch.float32, device=self.device)
                            if dist.is_initialized():
                                dist.broadcast(new_score_t, src=0)
                            new_score = new_score_t.item()

                            next_candidates_scores.append(new_score)
                            next_candidates_latents.append(
                                [candidate_prev_latents[k_idx].squeeze(0).clone()]
                            )

                        else:
                            # Final timestep, no backtracking
                            latent_model_input_bt = [candidate_prev_latents[k_idx].squeeze(0).clone()]
                            x0 = latent_model_input_bt

                            if offload_model:
                                self.model.cpu()
                                torch.cuda.empty_cache()

                            # ====== Modification: Only rank0 calculates Reward & logs ======
                            if self.rank == 0:
                                videos = self.vae.decode(x0)
                                new_score, score_details = reward_model(
                                    (
                                        (
                                            videos[0].unsqueeze(0)
                                            .permute(0, 2, 1, 3, 4)[:, :32:2, :, :, :]
                                            .squeeze(0) + 1
                                        ) / 2 * 255
                                    ).to(dtype=torch.uint8),
                                    input_prompt
                                )
                                with open(logging_file, 'a') as f:
                                    f.write(f'score_details: {score_details}\n')
                            else:
                                new_score = 0.0

                            # --- Similarly, broadcast for synchronization ---
                            new_score_t = torch.tensor([new_score], dtype=torch.float32, device=self.device)
                            if dist.is_initialized():
                                dist.broadcast(new_score_t, src=0)
                            new_score = new_score_t.item()

                            next_candidates_scores.append(new_score)
                            next_candidates_latents.append(
                                [candidate_prev_latents[k_idx].squeeze(0).clone()]
                            )
                
                # todo
                # from pathlib import Path
                # parent_dir = Path(logging_file).parent
                # torch.save(videos[0], f'{parent_dir}/inter_{i}.pth')

                # Beam pruning here
                candidates_latents = next_candidates_latents
                candidates_scores = next_candidates_scores

                if t != timesteps[-1]:
                    # Sort by score in descending order
                    sorted_indices = sorted(
                        range(len(candidates_scores)),
                        key=lambda i: candidates_scores[i],
                        reverse=True
                    )
                    # Logging only on rank0
                    if self.rank == 0:
                        with open(logging_file, 'a') as f:
                            f.write(f'sorted_indices: {sorted_indices}\n\n')

                    # Keep only the top beams
                    candidates_latents = [candidates_latents[i] for i in sorted_indices[:num_beams]]
                    candidates_scores = [candidates_scores[i] for i in sorted_indices[:num_beams]]
                else:
                    # Final timestep, no backtracking
                    sorted_indices = sorted(
                        range(len(candidates_scores)),
                        key=lambda i: candidates_scores[i],
                        reverse=True
                    )
                    if self.rank == 0:
                        with open(logging_file, 'a') as f:
                            f.write(f'sorted_indices: {sorted_indices}\n\n')
                    best_latents = candidates_latents[sorted_indices[0]]

                if t != timesteps[-1]:
                    sample_scheduler.do_step_index()

            # Final result decoding
            latents = best_latents
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        # Post-processing
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None