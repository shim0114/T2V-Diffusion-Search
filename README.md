<div align="center">

# Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search

[![arXiv](https://img.shields.io/badge/arXiv-2501.19252-b31b1b.svg)](https://arxiv.org/abs/2501.19252)
[![Google Site](https://img.shields.io/badge/website-site-blue)](https://sites.google.com/view/t2v-dlbs)

<!-- <img src="images/Presentation4.gif" width="400"> -->
<img src="./images/figure1.png" width="48%" alt="Figure 1"> <img src="./images/normalized_reward_cost_fig1-1.png" width="42%" alt="Figure 1">

</div>

As done in recent LLMs, we consider scaling test-time compute in text-to-video generation. **Diffusion Latent Beam Search** efficiently and robustly maximizes alignment rewards during inference.

We provide implementations for several state-of-the-art models, including **Latte**, **CogVideoX**, and **Wan 2.1** (comming soon).

## 🚀 Getting Started with Latte
### Install Libraries
Please use `./Dockerfile` to build docker image or install python libraries specified in this dockerfile.

### Download Weights
```
bash download_weight.sh
```

### Run Inference
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

## 🦋　Setup & Inference with CogVideoX

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

### Run Inference
We provide two configuration files in the `configs/${method}` directory. 
Below are examples of how to run inference with different settings:

#### 5B models
```
# No DLBS
python3 CogVideoX/sample.py --config CogVideoX/configs/kb1/static.yaml
# DLBS 
python3 CogVideoX/sample.py --config CogVideoX/configs/dlbs/static.yaml
# DLBS-LA 
python3 CogVideoX/sample.py --config CogVideoX/configs/dlbs_la/static.yaml
```

#### 2B models
```
# No DLBS
python3 CogVideoX/sample_2b.py --config CogVideoX/configs/kb1/static.yaml
# DLBS 
python3 CogVideoX/sample_2b.py --config CogVideoX/configs/dlbs/static.yaml
# DLBS-LA 
python3 CogVideoX/sample_2b.py --config CogVideoX/configs/dlbs_la/static.yaml
```

## 💜 Setup & Inference with Wan 2.1
Comming soon...


## 📚 Citation

```bibtex
@article{oshima2025inference,
  title     = {Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search},
  author    = {Yuta Oshima and Masahiro Suzuki and Yutaka Matsuo and Hiroki Furuta},
  journal   = {arXiv preprint arXiv:2501.19252},
  year      = {2025},
  url       = {https://arxiv.org/abs/2501.19252},
}
```

## 🙏 Acknowledgements

We sincerely thank those who have open-sourced their works including, but not limited to, the repositories below:

- https://github.com/huggingface/diffusers
- https://github.com/Vchitect/Latte 
- https://github.com/zai-org/CogVideo
- https://github.com/Wan-Video/Wan2.1
- https://github.com/Vchitect/VBench 
- https://github.com/AILab-CVC/VideoCrafter
- https://github.com/CIntellifusion/VideoDPO
