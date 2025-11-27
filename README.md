<div align="center">

# Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search

[![arXiv](https://img.shields.io/badge/arXiv-2501.19252-b31b1b.svg)](https://arxiv.org/abs/2501.19252)
[![Google Site](https://img.shields.io/badge/website-site-blue)](https://sites.google.com/view/t2v-dlbs)


<!-- <img src="images/Presentation4.gif" width="400"> -->
<img src="./images/figure1.png" width="48%" alt="Figure 1"> <img src="./images/normalized_reward_cost_fig1-1.png" width="42%" alt="Figure 1">

</div>

The remarkable progress in text-to-video diffusion models enables the generation of photorealistic videos, although the content of these generated videos often includes unnatural movement or deformation, reverse playback, and motionless scenes. Recently, an alignment problem has attracted huge attention, where we steer the output of diffusion models based on some measure of the content's goodness. Because there is a large room for improvement of perceptual quality along the frame direction, we should address which metrics we should optimize and how we can optimize them in the video generation. In this paper, we propose diffusion latent beam search with lookahead estimator, which can select a better diffusion latent to maximize a given alignment reward at inference time. We then point out that improving perceptual video quality with respect to alignment to prompts requires reward calibration by weighting existing metrics. This is because when humans or vision language models evaluate outputs, many previous metrics to quantify the naturalness of video do not always correlate with the evaluation. We demonstrate that our method improves the perceptual quality evaluated on the calibrated reward, VLMs, and human assessment, without model parameter update, and outputs the best generation compared to greedy search and best-of-N sampling under much more efficient computational cost. The experiments highlight that our method is beneficial to many capable generative models, and provide a practical guideline: we should prioritize the inference-time compute allocation into enabling the lookahead estimator and increasing the search budget, rather than expanding the denoising steps.

## Latte
### Install Libraries
Please use `./Dockerfile` to build docker image or install python libraries specified in this dockerfile.

### Download Weights
```
bash download_weight.sh
```

## Run Inference
We provide two configuration files in the `configs/${method}` directory. 
Below are examples of how to run inference with different settings:
```
# No DLBS 
python3 sample/sample_t2x.py --config configs/kb1/static.yaml
# DLBS 
python3 sample/sample_t2x.py --config configs/dlbs/static.yaml
# DLBS-LA 
python3 sample/sample_t2x.py --config configs/dlbs_la/static.yaml
```

## CogVideoX

### Install Libraries
```
cd CogVideoX/
pip3 install -r requirements.txt
cd ../
```

### Download Weights
```
bash download_weight.sh
cp -r pretrained CogVideoX
```

## Run Inference
We provide two configuration files in the `configs/${method}` directory. 
Below are examples of how to run inference with different settings:
```
# No DLBS 
python3 sample/sample_t2x.py --config configs/kb1/static.yaml
# DLBS 
python3 sample/sample_t2x.py --config configs/dlbs/static.yaml
# DLBS-LA 
python3 sample/sample_t2x.py --config configs/dlbs_la/static.yaml
```


## Citation

```bibtex
@article{oshima2025inference,
  title     = {Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search},
  author    = {Yuta Oshima and Masahiro Suzuki and Yutaka Matsuo and Hiroki Furuta},
  journal   = {arXiv preprint arXiv:2501.19252},
  year      = {2025},
  url       = {https://arxiv.org/abs/2501.19252},
}
```

## Acknowledgement

We sincerely thank those who have open-sourced their works including, but not limited to, the repositories below:

- https://github.com/huggingface/diffusers
- https://github.com/Vchitect/Latte 
- https://github.com/Vchitect/VBench 
