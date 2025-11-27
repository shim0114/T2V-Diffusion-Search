import os
import random
import numpy as np
import torch
import argparse
from omegaconf import OmegaConf
import time

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from my_pipeline import MyCogVideoXPipeline
from my_reward_model import VideoRewardCalculator
from my_scheduler import custom_step, mystep
import types

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    seed_everything(args.seed)

    # prompt = "1girl, 3d anime style, dancing, masterpiece."


    # pipe = CogVideoXPipeline.from_pretrained(
    #     "THUDM/CogVideoX-5b",
    #     torch_dtype=torch.bfloat16
    # )
    pipe = MyCogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.bfloat16
    )
    
    args.save_img_path = args.save_img_path.rstrip("/") + "_2B/"

    # Replace scheduler functions
    pipe.scheduler.step = types.MethodType(custom_step, pipe.scheduler)
    pipe.scheduler.mystep = types.MethodType(mystep, pipe.scheduler)

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    compute_video_reward = VideoRewardCalculator(
        w_subject_consistency=args.weights_list[0],
        w_motion_smoothness=args.weights_list[1],
        w_dynamic_degree=args.weights_list[2],
        w_aesthetic=args.weights_list[3],
        w_overall_consistency=args.weights_list[4],
    )

    if not os.path.exists(args.save_img_path):
            os.makedirs(args.save_img_path, exist_ok=True)
            
    num_beams = args.num_beams
    num_candidates = args.num_candidates
    
    for num_prompt, prompt in enumerate(args.text_prompt):
        print('Processing the ({}) prompt'.format(prompt))
        
        if not os.path.exists(args.save_img_path + f'/LA={args.num_backtrack_steps}_K={num_candidates}_B={num_beams}/'):
            os.makedirs(args.save_img_path + f'/LA={args.num_backtrack_steps}_K={num_candidates}_B={num_beams}/', exist_ok=True)

        if os.path.exists(f'{args.save_img_path}/LA={args.num_backtrack_steps}_K={num_candidates}_B={num_beams}/{prompt}_log.txt'):
            raise ValueError(f'File {args.save_img_path}/LA={args.num_backtrack_steps}_K={num_candidates}_B={num_beams}/{prompt}_log.txt already exists')

        start = time.time()
        
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.num_sampling_steps,
            num_frames=args.video_length, # Should be 8N + 1 where N <= 6 (default 49)
            guidance_scale=args.guidance_scale,
            eta=args.ddim_eta, 
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            num_beams=num_beams,
            num_candidates=num_candidates,
            num_backtrack_steps=args.num_backtrack_steps,
            reward_model=compute_video_reward,
            logging_file=f'{args.save_img_path}/LA={args.num_backtrack_steps}_K={num_candidates}_B={num_beams}/{prompt}_log.txt',
        ).frames[0]
        
        end = time.time()
        with open(f'{args.save_img_path}/LA={args.num_backtrack_steps}_K={num_candidates}_B={num_beams}/{prompt}_log.txt', 'a') as f:
            f.write(f'\nTime: {end-start}\n')
            
        export_to_video(video, f'{args.save_img_path}/LA={args.num_backtrack_steps}_K={num_candidates}_B={num_beams}/{prompt}-0000.mp4', fps=8)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/wbv10m_train.yaml")
    args = parser.parse_args()

    main(OmegaConf.load(args.config))