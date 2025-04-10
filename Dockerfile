# nvidia docker image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y sudo python3 python3-pip git ffmpeg wget curl vim unzip

RUN pip3 install --upgrade pip

# inatall python libraries
RUN pip3 install numpy pandas matplotlib scikit-learn jupyterlab wandb thop
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install timm diffusers[torch]==0.24.0 accelerate tensorboard einops transformers==4.44.2 av \
    scikit-image decord imageio-ffmpeg sentencepiece beautifulsoup4 ftfy omegaconf
RUN pip3 install imageio opencv-python 
# vbench
COPY verifiers/VBench/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
# reinstall
RUN pip3 install transformers==4.44.2