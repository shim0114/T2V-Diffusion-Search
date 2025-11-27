from diffusers import CogVideoXPipeline
from diffusers.utils import BaseOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from dataclasses import dataclass
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from my_reward_model import VideoRewardCalculator


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class MyCogVideoXPipeline(CogVideoXPipeline):
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        # DLBS params
        num_beams: int = 1,
        num_candidates: int = 1,
        num_backtrack_steps: int = 1,
        reward_model: VideoRewardCalculator = None, 
        logging_file: str = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_channels = self.transformer.config.in_channels
        
        candidates_scores = [0.0 for _ in range(num_beams)]
        candidates_latents = []
        for _ in range(num_beams):
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                latent_channels,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents=None,
            )
            candidates_latents.append(latents)
        # print(candidates_latents[0][0][0][0][0])
        # print(candidates_latents[1][0][0][0][0])
        # import sys
        # sys.exit()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                next_candidates_latents = []
                next_candidates_scores = []
                
                with open(logging_file, 'a') as f:
                    f.write(f'timestep: {t}\n')
                    
                if self.interrupt:
                    continue
                
                # Expand all beams in candidates_latents
                for b_idx, current_beam_latents in enumerate(candidates_latents):
                    
                    # Inference for current beam (Free2Guide, Algorithm 2, line 2-3)
                    latents = current_beam_latents.clone()  # copy to avoid modification of original latents
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    self._current_timestep = t

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])
                    # print(timestep)

                    # predict noise model_output
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.float()

                    # perform guidance
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)
                        base_latents = output[0][0]
                        base_original_latents = output[0][1]
                        base_latents_no_noise = output[1]
                    else:
                        # latents, old_pred_original_sample = self.scheduler.step(
                        #     noise_pred,
                        #     old_pred_original_sample,
                        #     t,
                        #     timesteps[i - 1] if i > 0 else None,
                        #     latents,
                        #     **extra_step_kwargs,
                        #     return_dict=False,
                        # )
                        raise NotImplementedError()
                    
                    for k_idx in range(num_candidates):
                        # If not the last step (t != timesteps[-1]), generate a sample with added noise and perform backtracking
                        if t != self.scheduler.timesteps[-1]:
                            # Calculate previous and next timesteps
                            prev_timestep = (
                                t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                            )
                            variance = self.scheduler._get_variance(t, prev_timestep)
                            std_dev_t = eta * variance ** 0.5

                            # Create a sample by adding Gaussian noise to base_latents_no_noise
                            latents = base_latents_no_noise + torch.randn_like(base_latents_no_noise) * std_dev_t
                            latents = latents.to(prompt_embeds.dtype)
                            
                            next_t = timesteps[i+1]
                            
                            latents_bt = latents.clone()
                            for bts in range(num_backtrack_steps, 0, -1):
                                t_bt = (next_t * bts) // num_backtrack_steps
                                t_next_bt = (next_t * (bts-1)) // num_backtrack_steps
                            
                                latent_model_input = torch.cat([latents_bt] * 2) if do_classifier_free_guidance else latents_bt
                                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_bt) 
                                

                                self._current_timestep = t_bt

                                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                                backtracking_timestep = t_bt.expand(latent_model_input.shape[0])
                                # print(backtracking_timestep)

                                # predict noise model_output
                                noise_pred = self.transformer(
                                    hidden_states=latent_model_input,
                                    encoder_hidden_states=prompt_embeds,
                                    timestep=backtracking_timestep,
                                    image_rotary_emb=image_rotary_emb,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                )[0]
                                noise_pred = noise_pred.float()

                                # perform guidance
                                if use_dynamic_cfg:
                                    self._guidance_scale = 1 + guidance_scale * (
                                        (1 - math.cos(math.pi * ((num_inference_steps - t_bt.item()) / num_inference_steps) ** 5.0)) / 2
                                    )
                                if do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                                # compute the previous noisy sample x_t -> x_t-1
                                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                                    output = self.scheduler.mystep(noise_pred, t_bt, t_next_bt, latents_bt, eta=0.0, return_dict=False)
                                    latents_bt = output[0][0].to(prompt_embeds.dtype)
                                    original_latents = output[0][1].to(prompt_embeds.dtype)
                                else:
                                    # latents, old_pred_original_sample = self.scheduler.step(
                                    #     noise_pred,
                                    #     old_pred_original_sample,
                                    #     t,
                                    #     timesteps[i - 1] if i > 0 else None,
                                    #     latents,
                                    #     **extra_step_kwargs,
                                    #     return_dict=False,
                                    # )
                                    raise NotImplementedError()
                            
                        else:
                            latents = base_original_latents.to(prompt_embeds.dtype) # base_latents
                            original_latents = base_original_latents.to(prompt_embeds.dtype)
                           
                        # reward calculation 
                        if not output_type == "latent":
                            # Discard any padding frames that were added for CogVideoX 1.5
                            original_latents = original_latents[:, additional_frames:]
                            original_video = self.decode_latents(original_latents)
                            # video = self.video_processor.postprocess_video(video=original_video, output_type=output_type)
                            # from diffusers.utils import export_to_video
                            # export_to_video(video[0], f"output_dlbs_{i}.mp4", fps=8)
                        else:
                            video = original_latents
                    
                        next_candidates_latents.append(latents.clone())
                        # print(original_video.min(), original_video.max())
                        new_score, score_details = reward_model(
                            (((original_video.permute(0, 2, 1, 3, 4)[:, :16, :, :, :].squeeze(0) + 1.) / 2.) * 255).to(dtype=torch.uint8), 
                            prompt
                            )
                        next_candidates_scores.append(new_score)
                        
                        with open(logging_file, 'a') as f:
                            f.write(f'score_details: {score_details}\n')
                    

                # call the callback, if provided
                if callback_on_step_end is not None:
                    raise NotImplementedError()
                    # callback_kwargs = {}
                    # for k in callback_on_step_end_tensor_inputs:
                    #     callback_kwargs[k] = locals()[k]
                    # callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    # latents = callback_outputs.pop("latents", latents)
                    # prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    # negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    
                candidates_latents = next_candidates_latents
                candidates_scores = next_candidates_scores
                
                # If not the last timestep, perform beam pruning
                if not t == timesteps[-1]:
                    # Sort by score in descending order
                    sorted_indices = sorted(range(len(candidates_scores)), key=lambda i: candidates_scores[i], reverse=True)
                    with open(logging_file, 'a') as f:
                        f.write(f'sorted_indices: {sorted_indices}\n')
                        f.write('\n')
                    # Keep only the top beams
                    candidates_latents = [candidates_latents[i] for i in sorted_indices[:num_beams]]
                    candidates_scores = [candidates_scores[i] for i in sorted_indices[:num_beams]]
                else:
                    # For the final step, select the best beam after scoring
                    sorted_indices = sorted(range(len(candidates_scores)), key=lambda i: candidates_scores[i], reverse=True)
                    with open(logging_file, 'a') as f:
                        f.write(f'sorted_indices: {sorted_indices}\n')
                        f.write('\n')
                    best_latents = candidates_latents[sorted_indices[0]]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            best_latents = best_latents[:, additional_frames:]
            video = self.decode_latents(best_latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = best_latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
    