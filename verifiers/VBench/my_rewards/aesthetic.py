import os
import sys

sys.path.append(os.path.abspath("verifiers/VBench"))

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from urllib.request import urlretrieve
from vbench.utils import load_video, load_dimension_info, clip_transform
from tqdm import tqdm


# from .distributed import (
#     get_world_size,
#     get_rank,
#     all_gather,
#     barrier,
#     distribute_list_to_rank,
#     gather_list_of_dict,
# )

def get_aesthetic_model(cache_folder):
    """load the aethetic model"""
    path_to_model = cache_folder + "/sa_0_4_vit_l_14_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        )
        # download aesthetic predictor
        if not os.path.isfile(path_to_model):
            try:
                print(f'trying urlretrieve to download {url_model} to {path_to_model}')
                urlretrieve(url_model, path_to_model) # unable to download https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true to pretrained/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth 
            except:
                print(f'unable to download {url_model} to {path_to_model} using urlretrieve, trying wget')
                wget_command = ['wget', url_model, '-P', os.path.dirname(path_to_model)]
                subprocess.run(wget_command)
    m = nn.Linear(768, 1)
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def laion_aesthetic(aesthetic_model, clip_model, video_tensor, device, batch_size=32):
    aesthetic_model.eval()
    clip_model.eval()
    image_transform = clip_transform(224)
    
    images = video_tensor

    for i in range(0, len(images), batch_size):
        image_batch = images[i:i + batch_size]
        image_batch = image_transform(image_batch)
        image_batch = image_batch.to(device)

        with torch.no_grad():
            image_feats = clip_model.encode_image(image_batch).to(torch.float32)
            image_feats = F.normalize(image_feats, dim=-1, p=2)
            aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)

    normalized_aesthetic_scores = aesthetic_scores / 10
    cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)
    return cur_avg.item()
    # aesthetic_avg += cur_avg.item()
    # num += 1
    # video_results.append({'video_path': video_path, 'video_results': cur_avg.item()})

    # aesthetic_avg /= num
    # return aesthetic_avg, video_results

def compute_aesthetic_quality(video_tensor, device, submodules_list, **kwargs):
    vit_path = submodules_list[0]
    aes_path = submodules_list[1]
    # if get_rank() == 0:
    #     aesthetic_model = get_aesthetic_model(aes_path).to(device)
    #     barrier()
    # else:
    #     barrier()
    #     aesthetic_model = get_aesthetic_model(aes_path).to(device)
    clip_model, preprocess = clip.load(vit_path, device=device)
    aesthetic_model = get_aesthetic_model(aes_path).to(device)
    # video_list, _ = load_dimension_info(json_dir, dimension='aesthetic_quality', lang='en')
    # video_list = distribute_list_to_rank(video_list)
    # all_results, video_results = laion_aesthetic(aesthetic_model, clip_model, video_list, device)
    aesthetic_score = laion_aesthetic(aesthetic_model, clip_model, video_tensor, device)
    return aesthetic_score
    # if get_world_size() > 1:
    #     video_results = gather_list_of_dict(video_results)
    #     all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    # return all_results, video_results
    

if __name__ == '__main__':
    # json_dir = 'aesthetic_quality.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    submodules_list = ['ViT-L/14', 'pretrained/aesthetic_model/emb_reader']
    video_tensor = load_video('results_sampling/a cat wearing sunglasses and working as a lifeguard at pool.-0000.mp4') # results_sampling/best-of-n/devil_video_low_64/Old man standing up with his hand on heart during the national anthem at an international sport event or football competition. close-up.-0000.mp4')
    # all_results, video_results = compute_aesthetic_quality(json_dir, device, submodules_list)
    aesthetic_score = compute_aesthetic_quality(video_tensor, device, submodules_list)
    print(aesthetic_score)
    