wget -P "pretrained/amt_model" "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth"

wget -P "pretrained/raft_model" "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
unzip -d "pretrained/raft_model" "pretrained/raft_model/models.zip"
rm -r "pretrained/raft_model/models.zip"

wget -P "pretrained/ViCLIP" "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"
wget -P "pretrained/ViCLIP" "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
