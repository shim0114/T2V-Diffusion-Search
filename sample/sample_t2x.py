import os
import torch
import argparse
import torchvision

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from transformers import T5EncoderModel, T5Tokenizer

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
from my_utils import save_video_grid
import imageio
from torchvision.utils import save_image

from dlbs_scheduler import MyDDIMScheduler

from dlbs_pipeline_latte import VideoRewardCalculator
from dlbs_pipeline_latte import LattePipeline as DLBSLattePipeline

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer_model = get_models(args).to(device, dtype=torch.float16)
    
    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DLBS': 
        scheduler = MyDDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start, 
                                                  beta_end=args.beta_end, 
                                                  beta_schedule=args.beta_schedule,
                                                  variance_type=args.variance_type,
                                                  clip_sample=False)
        
        videogen_pipeline = DLBSLattePipeline(vae=vae, 
                                 text_encoder=text_encoder, 
                                 tokenizer=tokenizer, 
                                 scheduler=scheduler, 
                                 transformer=transformer_model).to(device)
        
    else:
        raise NotImplementedError()


    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    # Reward calculator
    compute_video_reward = VideoRewardCalculator(
        device=args.device_r, 
        w_subject_consistency=args.weights_list[0],
        w_motion_smoothness=args.weights_list[1],
        w_dynamic_degree=args.weights_list[2],
        w_aesthetic=args.weights_list[3],
        w_overall_consistency=args.weights_list[4],
    )
    
    for num_prompt, prompt in enumerate(args.text_prompt):
        print('Processing the ({}) prompt'.format(prompt))
        
        if not os.path.exists(args.save_img_path + f'/LA={args.num_lookahead_steps}_K={args.num_candidates}_B={args.num_beams}/'):
            os.makedirs(args.save_img_path + f'/LA={args.num_lookahead_steps}_K={args.num_candidates}_B={args.num_beams}/', exist_ok=True)

        if os.path.exists(f'{args.save_img_path}/LA={args.num_lookahead_steps}_K={args.num_candidates}_B={args.num_beams}/{prompt}_log.txt'):
            raise ValueError(f'File {args.save_img_path}/LA={args.num_lookahead_steps}_K={args.num_candidates}_B={args.num_beams}/{prompt}_log.txt already exists')
            
        videos = videogen_pipeline(prompt, 
                                video_length=args.video_length, 
                                height=args.image_size[0], 
                                width=args.image_size[1], 
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale,
                                enable_temporal_attentions=args.enable_temporal_attentions,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                enable_vae_temporal_decoder=args.enable_vae_temporal_decoder,
                                eta=args.ddim_eta,
                                num_beams=args.num_beams,
                                num_candidates=args.num_candidates,
                                num_lookahead_steps=args.num_lookahead_steps,
                                reward_model=compute_video_reward,
                                logging_file=f'{args.save_img_path}/LA={args.num_lookahead_steps}_K={args.num_candidates}_B={args.num_beams}/{prompt}_log.txt'
                                ).video

        if videos.shape[1] == 1:
            try:
                save_image(videos[0][0], args.save_img_path + f'/LA={args.num_lookahead_steps}_K={args.num_candidates}_B={args.num_beams}/' + prompt + '.png')
            except:
                # save_image(videos[0][0], args.save_img_path + str(num_prompt)+ '.png')
                print('Error when saving {}'.format(prompt))
        else:
            try:
                imageio.mimwrite(args.save_img_path + f'/LA={args.num_lookahead_steps}_K={args.num_candidates}_B={args.num_beams}/' + prompt + '.mp4', videos[0], fps=8, quality=9) # highest quality is 10, lowest is 0
            except:
                print('Error when saving {}'.format(prompt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/wbv10m_train.yaml")
    args = parser.parse_args()

    main(OmegaConf.load(args.config))