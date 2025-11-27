import os
import json
import numpy as np
import sys
sys.path.append(os.path.abspath("verifiers/VBench"))

import torch
import clip
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, clip_transform, read_frames_decord_by_fps, CACHE_DIR
from vbench.third_party.ViCLIP.viclip import ViCLIP
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer

# from .distributed import (
#     get_world_size,
#     get_rank,
#     all_gather,
#     barrier,
#     distribute_list_to_rank,
#     gather_list_of_dict,
# )


def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]
    text_template= input_text # f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)      
        text_feature_dict[input_text] = text_features
    return text_features

def get_vid_features(model, input_frames):
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames,test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)    
    return clip_feat

def get_predict_label(clip_feature, text_feats_tensor, top=5):
    label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
    return top_probs, top_labels

def overall_consistency(clip_model, video_tensor, prompt, tokenizer, device, sample="middle"):
    # sim = []
    # video_results = []
    image_transform = clip_transform(224)
    # for info in tqdm(video_dict, disable=get_rank() > 0):
        # query = info['prompt']
        # # text = clip.tokenize([query]).to(device)
        # video_list = info['video_list']
        # for video_path in video_list:
    query = prompt
    # cur_video = []
    with torch.no_grad():
        # images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
        if len(video_tensor) != 16:
            raise NotImplementedError("This function is implemented for videos with 2 fps.")
        
        # get frame_indices
        # frame_indices = np.random.permutation(16)[:8]
        # frame_indices.sort()
        # frame_indices = list(frame_indices)
        
        images = video_tensor[::2] # [frame_indices]
        images = image_transform(images)
        images = images.to(device)
        clip_feat = get_vid_features(clip_model,images.unsqueeze(0))
        text_feat = get_text_features(clip_model, query, tokenizer)
        logit_per_text =  clip_feat @ text_feat.T
        score_per_video =  float(logit_per_text[0][0].cpu())
        # sim.append(score_per_video)
        # video_results.append({'video_path': video_path, 'video_results': score_per_video})
    # avg_score = np.mean(sim)
    return score_per_video # avg_score, video_results

def compute_overall_consistency(video_tensor, prompt, device, submodules_list, **kwargs):
    CACHE_DIR = 'pretrained'
    tokenizer = SimpleTokenizer(f"{CACHE_DIR}/ViCLIP/bpe_simple_vocab_16e6.txt.gz")
    viclip = ViCLIP(tokenizer= tokenizer, **submodules_list).to(device)
    # _, video_dict = load_dimension_info(json_dir, dimension='overall_consistency', lang='en')
    # video_dict = distribute_list_to_rank(video_dict)
    # video_tensor = load_video('results_sampling/best-of-n/devil_video_low_64/Old man standing up with his hand on heart during the national anthem at an international sport event or football competition. close-up.-0001.mp4')
    # prompt = "Old man standing up with his hand on heart during the national anthem at an international sport event or football competition. close-up."
    overall_consistency_score = overall_consistency(viclip, video_tensor, prompt, tokenizer, device)
    # all_results, video_results = overall_consistency(viclip, video_tensor, tokenizer, device)
    # if get_world_size() > 1:
    #     video_results = gather_list_of_dict(video_results)
    #     all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return overall_consistency_score # all_results, video_results

if __name__ == '__main__':
    # json_dir = 'overall_consistency.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CACHE_DIR = 'pretrained'
    submodules_list = {"pretrain": f'{CACHE_DIR}/ViCLIP/ViClip-InternVid-10M-FLT.pth'}
    wget_command_viclip = ['wget', 'https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth', '-P', f'{CACHE_DIR}/ViCLIP']
    wget_command_tokenizer = ['wget', 'https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz', '-P', f'{CACHE_DIR}/ViCLIP']
    # import subprocess
    # subprocess.run(wget_command_viclip)  
    # subprocess.run(wget_command_tokenizer)
    video_tensor = load_video('test-videos/high/A boat hits a big wave and flips, landing upside down.-0000.mp4')
    prompt = "A boat hits a big wave and flips, landing upside down."
    overall_consistency_score = compute_overall_consistency(video_tensor, prompt, device, submodules_list)
    print(overall_consistency_score)