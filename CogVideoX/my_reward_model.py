import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("verifiers/VBench"))
import clip
from verifiers.VBench.my_rewards.aesthetic import get_aesthetic_model, laion_aesthetic
from verifiers.VBench.my_rewards.dino_similarity import subject_consistency
from easydict import EasyDict as edict
from verifiers.VBench.my_rewards.optical_flow import DynamicDegree, dynamic_degree
from vbench.third_party.ViCLIP.viclip import ViCLIP
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
from verifiers.VBench.my_rewards.viclip_similarity import overall_consistency
from verifiers.VBench.my_rewards.motion_prior import MotionSmoothness, motion_smoothness

def dd_mapping_func(a):
    return np.log(a) / 16.0
    
class VideoRewardCalculator:
    """
    This class initializes models and weights, 
    and calculates an overall reward score 
    based on the input video_tensor and prompt using compute_reward()
    """

    def __init__(
        self,
        device='cuda:0',
        w_subject_consistency=1.0,
        w_motion_smoothness=1.0,
        w_dynamic_degree=1.0,
        w_aesthetic=1.0,
        w_overall_consistency=1.0,
    ):
        """
        Parameters
        ----------
        device : str
            PyTorch execution device
        w_subject_consistency : float
            subject_consistency score weight
        w_motion_smoothness : float
            motion_smoothness score weight
        w_dynamic_degree : float
            dynamic_degree score weight
        w_aesthetic : float
            aesthetic score weight
        w_overall_consistency : float
            overall_consistency score weight
        """

        self.device = device
        self.w_subject_consistency = w_subject_consistency
        self.w_motion_smoothness = w_motion_smoothness
        self.w_dynamic_degree = w_dynamic_degree
        self.w_aesthetic = w_aesthetic
        self.w_overall_consistency = w_overall_consistency

        # ------- Initialize models --------
        # aesthetic
        self.clip_model, self.preprocess = clip.load('ViT-L/14', device=self.device)
        self.aesthetic_model = get_aesthetic_model('pretrained/aesthetic_model/emb_reader').to(self.device)

        # dino
        dino_submodules_list = {
            'repo_or_dir': 'facebookresearch/dino:main',
            'source': 'github',
            'model': 'dino_vitb16',
            'read_frame': False,
        }
        self.dino_model = torch.hub.load(**dino_submodules_list).to(self.device)

        # raft -> DynamicDegree
        args_new = edict({"model": 'pretrained/raft_model/models/raft-things.pth',
                          "small": False,
                          "mixed_precision": False,
                          "alternate_corr": False})
        self.dynamic = DynamicDegree(args_new, self.device)  

        # ViCLIP
        self.tokenizer = SimpleTokenizer("pretrained/ViCLIP/bpe_simple_vocab_16e6.txt.gz")
        viclip_submodules_list = {"pretrain": 'pretrained/ViCLIP/ViClip-InternVid-10M-FLT.pth'}
        self.viclip = ViCLIP(tokenizer=self.tokenizer, **viclip_submodules_list).to(self.device)

        # amt -> MotionSmoothness
        self.motion = MotionSmoothness('verifiers/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml', 
                                       'pretrained/amt_model/amt-s.pth', 
                                       self.device)

    @torch.no_grad()
    def __call__(self, video_tensor, prompt, image_reward=False):
        """
        Calculates various scores from the specified video_tensor and prompt,
        returning a reward value based on a weighted sum, along with details of each score.

        Parameters
        ----------
        video_tensor : torch.Tensor
            Video data tensor (expected shape such as (B, T, C, H, W)).
        prompt : str
            Prompt input for overall_consistency.

        Returns
        -------
        reward : float
            Reward calculated as a weighted linear sum of the 5 scores.
        score_details : dict
            Dictionary summarizing the details of each score.
        """
        video_tensor = video_tensor.to(self.device)

        # ---- Calculate each score ----
        aesthetic_score = laion_aesthetic(
            self.aesthetic_model, self.clip_model, video_tensor, self.device
        ) 

        subject_consistency_score = subject_consistency(
            self.dino_model, video_tensor, self.device
        ) if not image_reward else 0.0

        dynamic_degree_score = dd_mapping_func(
            dynamic_degree(
                self.dynamic, video_tensor.to(torch.float)
            )
        )if not image_reward else 0.0

        overall_consistency_score = overall_consistency(
            self.viclip, video_tensor, prompt, self.tokenizer, self.device
        ) if not image_reward else 0.0
        
        motion_smoothness_score = motion_smoothness(
            self.motion, video_tensor
        ) if not image_reward else 0.0

        # ---- Calculate weighted sum ----
        reward = (
            self.w_aesthetic           * aesthetic_score +
            self.w_subject_consistency * subject_consistency_score +
            self.w_dynamic_degree      * dynamic_degree_score +  
            self.w_overall_consistency * overall_consistency_score +
            self.w_motion_smoothness   * motion_smoothness_score
        )

        score_details = {
            'reward'              : reward,
            'subject_consistency' : subject_consistency_score,
            'motion_smoothness'   : motion_smoothness_score,
            'dynamic_degree'      : dynamic_degree_score,
            'aesthetic'           : aesthetic_score,
            'overall_consistency' : overall_consistency_score
        }

        return float(reward), score_details
