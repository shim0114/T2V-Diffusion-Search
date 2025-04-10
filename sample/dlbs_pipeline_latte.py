# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union

import time

import torch
import einops
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, Transformer2DModel
# from diffusers.schedulers import DPMSolverMultistepScheduler
### Changed ###
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.utils import (
    BACKENDS_MAPPING,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from dataclasses import dataclass

# import hpsv2
import imageio
from PIL import Image
import cv2
import numpy as np

import time
import threading

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

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PixArtAlphaPipeline

        >>> # You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
        >>> pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
        >>> # Enable memory optimizations.
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "A small cactus with a happy face in the Sahara desert."
        >>> image = pipe(prompt).images[0]
        ```
"""

@dataclass
class VideoPipelineOutput(BaseOutput):
    video: torch.Tensor
    
def dd_mapping_func(a):
    return np.log(a) / 16.0
    
class VideoRewardCalculator:
    """
    Video Reward Calculator for evaluating video generation quality.
    """

    def __init__(
        self,
        device='cuda',
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
            Device name. 
        w_subject_consistency : float
            Weight for subject_consistency score
        w_motion_smoothness : float
            Weight for motion_smoothness score
        w_dynamic_degree : float
            Weight for dynamic_degree score
        w_aesthetic : float
            Weight for aesthetic score
        w_overall_consistency : float
            Weight for overall_consistency score
        """

        self.device = device
        self.w_subject_consistency = w_subject_consistency
        self.w_motion_smoothness = w_motion_smoothness
        self.w_dynamic_degree = w_dynamic_degree
        self.w_aesthetic = w_aesthetic
        self.w_overall_consistency = w_overall_consistency

        # init models
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

    def __call__(self, video_tensor, prompt, image_reward=False):
        """_summary_

        Args:
            video_tensor (_type_): _description_
            prompt (_type_): _description_
            image_reward (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        video_tensor = video_tensor.to(self.device)

        # calculate each score
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

        # calculate reward
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



class LattePipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using PixArt-Alpha.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. PixArt-Alpha uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`Transformer2DModel`]):
            A text conditioned `Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """
    bad_punct_regex = re.compile(
        r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: Transformer2DModel,
        scheduler: DPMSolverMultistepScheduler, ### changed ###
        # scheduler_ode: DPMSolverMultistepScheduler, ### changed ###
    ):
        super().__init__()
        
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/utils.py
    def mask_text_embeddings(self, emb, mask):
        if emb.shape[0] == 1:
            keep_index = mask.sum().item()
            return emb[:, :, :keep_index, :], keep_index # 1, 120, 4096 -> 1 7 4096
        else:
            masked_feature = emb * mask[:, None, :, None] # 1 120 4096
            return masked_feature, emb.shape[2]

    # Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        mask_feature: bool = True,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
                string.
            clean_caption (bool, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            mask_feature: (bool, defaults to `True`):
                If `True`, the function will mask the text embeddings.
        """
        embeds_initially_provided = prompt_embeds is not None and negative_prompt_embeds is not None

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = 120

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds_attention_mask = attention_mask

            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds_attention_mask = torch.ones_like(prompt_embeds)

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_embeds_attention_mask = prompt_embeds_attention_mask.view(bs_embed, -1)
        prompt_embeds_attention_mask = prompt_embeds_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
        else:
            negative_prompt_embeds = None

        # Perform additional masking.
        if mask_feature and not embeds_initially_provided:
            prompt_embeds = prompt_embeds.unsqueeze(1)
            masked_prompt_embeds, keep_indices = self.mask_text_embeddings(prompt_embeds, prompt_embeds_attention_mask)
            masked_prompt_embeds = masked_prompt_embeds.squeeze(1)
            masked_negative_prompt_embeds = (
                negative_prompt_embeds[:, :keep_indices, :] if negative_prompt_embeds is not None else None
            )

            # import torch.nn.functional as F

            # padding = (0, 0, 0, 113)  # (左, 右, 下, 上)
            # masked_prompt_embeds_ = F.pad(masked_prompt_embeds, padding, "constant", 0)
            # masked_negative_prompt_embeds_ = F.pad(masked_negative_prompt_embeds, padding, "constant", 0)

            # print(masked_prompt_embeds == masked_prompt_embeds_[:, :masked_negative_prompt_embeds.shape[1], ...])

            return masked_prompt_embeds, masked_negative_prompt_embeds
            # return masked_prompt_embeds_, masked_negative_prompt_embeds_
        
        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_steps,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warn(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warn(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        video_length: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        mask_feature: bool = True,
        enable_temporal_attentions: bool = True,
        enable_vae_temporal_decoder: bool = False,
        num_beams: int = 1,
        num_candidates: int = 1,
        num_lookahead_steps: int = 1,
        search_range: list = [1.0, 0.0],
        reward_model: VideoRewardCalculator = None,
        logging_file: str = None,
    ) -> Union[VideoPipelineOutput, Tuple]:
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
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
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
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            mask_feature (`bool` defaults to `True`): If set to `True`, the text embeddings will be masked.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, negative_prompt, callback_steps, prompt_embeds, negative_prompt_embeds
        )

        # 2. Default height and width to transformer
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
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
            mask_feature=mask_feature,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        
        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        
        candidates_scores = [0.0 for _ in range(num_beams)]
        candidates_latents = []
        for _ in range(num_beams):
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                latent_channels,
                video_length,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents=None,
            )
            candidates_latents.append(latents)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
          

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                next_candidates_latents = []
                next_candidates_scores = []
                
                with open(logging_file, 'a') as f:
                    f.write(f'timestep: {t}\n')

                for b_idx, current_beam_latents in enumerate(candidates_latents):
                    
                    latents = current_beam_latents.clone()  
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    current_timestep = t
                    if not torch.is_tensor(current_timestep):
                        is_mps = latent_model_input.device.type == "mps"
                        if isinstance(current_timestep, float):
                            dtype = torch.float32 if is_mps else torch.float64
                        else:
                            dtype = torch.int32 if is_mps else torch.int64
                        current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                    elif len(current_timestep.shape) == 0:
                        current_timestep = current_timestep[None].to(latent_model_input.device)
                    current_timestep = current_timestep.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=current_timestep,
                        added_cond_kwargs=added_cond_kwargs,
                        enable_temporal_attentions=enable_temporal_attentions,
                        return_dict=False,
                    )[0]

                    # classifier-free guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # learned sigma check
                    if self.transformer.config.out_channels // 2 == latent_channels:
                        noise_pred = noise_pred.chunk(2, dim=1)[0]

                    output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                    base_latents = output[0]["prev_sample"]
                    base_original_latents = output[0]["pred_original_sample"]
                    base_latents_no_noise = output[1]
                    
                    for k_idx in range(num_candidates):
                        if not t == self.scheduler.timesteps[-1]:
                            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                            variance = self.scheduler._get_variance(t, prev_timestep)
                            std_dev_t = eta * variance ** (0.5)
                            
                            latents = base_latents_no_noise + \
                                torch.randn_like(base_latents_no_noise) * std_dev_t
                            
                            next_timestep = timesteps[i+1] 
                            
                            
                            latents_bt = latents.clone()
                            for bts in range(num_lookahead_steps, 0, -1):
                                t_bt = (next_timestep * bts) // num_lookahead_steps
                                t_next_bt = (next_timestep * (bts-1)) // num_lookahead_steps
                                
                                latent_model_input = torch.cat([latents_bt] * 2) if do_classifier_free_guidance else latents_bt
                                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_bt)

                                lookahead_timestep = t_bt
                                if not torch.is_tensor(lookahead_timestep):
                                    is_mps = latent_model_input.device.type == "mps"
                                    if isinstance(lookahead_timestep, float):
                                        dtype = torch.float32 if is_mps else torch.float64
                                    else:
                                        dtype = torch.int32 if is_mps else torch.int64
                                    lookahead_timestep = torch.tensor([lookahead_timestep], dtype=dtype, device=latent_model_input.device)
                                elif len(lookahead_timestep.shape) == 0:
                                    lookahead_timestep = lookahead_timestep[None].to(latent_model_input.device)
                                lookahead_timestep = lookahead_timestep.expand(latent_model_input.shape[0])

                                noise_pred = self.transformer(
                                    latent_model_input,
                                    encoder_hidden_states=prompt_embeds,
                                    timestep=lookahead_timestep,
                                    added_cond_kwargs=added_cond_kwargs,
                                    enable_temporal_attentions=enable_temporal_attentions,
                                    return_dict=False,
                                )[0]

                                # classifier-free guidance
                                if do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                                # learned sigma check
                                if self.transformer.config.out_channels // 2 == latent_channels:
                                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                                output = self.scheduler.mystep(noise_pred, t_bt, t_next_bt, latents_bt, eta=0.0, return_dict=True)
                                latents_bt = output[0]["prev_sample"]
                                original_latents = output[0]["pred_original_sample"]
                        
                        else:
                            latents = base_original_latents # base_latents
                            original_latents = base_original_latents
                        
                        # Evaluate candidates with reward model
                        if not output_type == 'latents':      
                            if latents.shape[2] == 1: # image
                                original_video = self.decode_latents_image(original_latents)
                            else: # video
                                if enable_vae_temporal_decoder:
                                    original_video = self.decode_latents_with_temporal_decoder(original_latents)
                                else:
                                    original_video = self.decode_latents(original_latents)
                        else:
                            original_video = original_latents

                        # if not os.path.exists(f"results_sampling/intermediate/bt{num_lookahead_steps}/{search_range[0]}_{search_range[1]}"):
                        #     os.makedirs(f"results_sampling/intermediate/bt{num_lookahead_steps}/{search_range[0]}_{search_range[1]}")
                        # imageio.mimwrite(f'results_sampling/intermediate/bt{num_lookahead_steps}/{search_range[0]}_{search_range[1]}/{prompt}_{t}_{b_idx}_{k_idx}.mp4', original_video[0], fps=8, quality=9)
                        
                        if latents.shape[2] == 1: # image
                            next_candidates_latents.append(latents.clone())
                            original_video = (original_video.permute(0, 1, 3, 4, 2)* 255).to(dtype=torch.uint8)
                            new_score, score_details = reward_model(
                                original_video.permute(0, 1, 4, 2, 3).squeeze(0), 
                                prompt,
                                image_reward=True
                                )
                            next_candidates_scores.append(new_score)
                        else:
                            next_candidates_latents.append(latents.clone())
                            new_score, score_details = reward_model(
                                original_video.permute(0, 1, 4, 2, 3).squeeze(0), 
                                prompt
                                )
                            next_candidates_scores.append(new_score)
                        
                        with open(logging_file, 'a') as f:
                            f.write(f'score_details: {score_details}\n')
                        
                        if t >= 1000 * search_range[0] or t < 1000 * search_range[1]:
                            break
                        
                    if t >= 1000 * search_range[0] or t < 1000 * search_range[1]:
                        if t >= 1000 * search_range[0] and prev_timestep < 1000 * search_range[0]:
                            # prepare beams for beam search
                            while len(next_candidates_latents) < num_beams:
                                variance = self.scheduler._get_variance(t, prev_timestep)
                                std_dev_t = eta * variance ** (0.5)
                                
                                latents = base_latents_no_noise + \
                                    torch.randn_like(base_latents_no_noise) * std_dev_t
                                    
                                next_candidates_latents.append(latents.clone())
                                next_candidates_scores.append(next_candidates_scores[0])            
                        break

                candidates_latents = next_candidates_latents
                candidates_scores = next_candidates_scores
                
                if not t == self.scheduler.timesteps[-1]:
                    # Sort candidates
                    sorted_indices = sorted(range(len(candidates_scores)), key=lambda i: candidates_scores[i], reverse=True)
                    with open(logging_file, 'a') as f:
                        f.write(f'sorted_indices: {sorted_indices}\n')
                        f.write('\n')
                    # Select top-k candidates
                    candidates_latents = [candidates_latents[i] for i in sorted_indices[:num_beams]]
                    candidates_scores = [candidates_scores[i] for i in sorted_indices[:num_beams]]
                else:
                    # Last timestep: select the best candidate
                    sorted_indices = sorted(range(len(candidates_scores)), key=lambda i: candidates_scores[i], reverse=True)
                    with open(logging_file, 'a') as f:
                        f.write(f'sorted_indices: {sorted_indices}\n')
                        f.write('\n')
                    best_latents = candidates_latents[sorted_indices[0]]

                # callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, best_latents if t == self.scheduler.timesteps[-1] else candidates_latents[0])

        if not output_type == 'latents':
            if best_latents.shape[2] == 1: # image
                video = self.decode_latents_image(best_latents)
            else: # video
                if enable_vae_temporal_decoder:
                    video = self.decode_latents_with_temporal_decoder(best_latents)
                else:
                    video = self.decode_latents(best_latents)
        else:
            video = best_latents
            return VideoPipelineOutput(video=video)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return VideoPipelineOutput(video=video)
    
    def decode_latents_image(self, latents):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = einops.rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(
                latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = einops.rearrange(video, "(b f) c h w -> b f c h w", f=video_length)
        video = (video / 2.0 + 0.5).clamp(0, 1)
        return video
    
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = einops.rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(
                latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = einops.rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        return video
    
    def decode_latents_with_temporal_decoder(self, latents):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = einops.rearrange(latents, "b c f h w -> (b f) c h w")
        video = []

        decode_chunk_size = 14
        for frame_idx in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[frame_idx : frame_idx + decode_chunk_size].shape[0]

            decode_kwargs = {}
            decode_kwargs["num_frames"] = num_frames_in

            video.append(self.vae.decode(latents[frame_idx:frame_idx+decode_chunk_size], **decode_kwargs).sample)
            
        video = torch.cat(video)
        video = einops.rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        return video
