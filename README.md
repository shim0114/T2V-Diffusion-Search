# T2V-Diffusion-Search

## Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search

[![arXiv](https://img.shields.io/badge/arXiv-2501.19252-b31b1b.svg)](https://arxiv.org/abs/2501.19252)
[![Google Site](https://img.shields.io/badge/website-site-blue)](https://sites.google.com/view/t2v-dlbs)

<!-- <img src="images/Presentation4.gif" width="400"> -->
<p align="center">
    <img src="images/search_qualitative-1.png" alt="Image1" width="800" />
</p>

## How to Work
Diffusion latent beam search (DLBS) seeks a better diffusion path over the reverse process; sampling K latents per beam and possessing B beams for the next step, which mitigates the effect from inaccurate argmax. Lookahead estimator notably reduces the noise at latent reward evaluation by interpolating the rest of the time steps from the current latent with deterministic DDIM.
<p align="center">
    <img src="images/figure1.png" alt="Image2" width="600" />
</p>

## Settings
### Install Libraries
Please use `./Dockerfile` to build docker image or install python libraries specified in this dockerfile.

### Download Weights
```
bash download_weight.sh
```

## Run Inference
```
# DLBS
python3 sample/sample_t2x.py --config configs/config_dlbs.yaml
# DLBS-LA
python3 sample/sample_t2x.py --config configs/config_dlbs_la.yaml
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