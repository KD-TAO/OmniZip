
# OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models

[Keda Tao](), [Kele Shao](), [Bohan Yu](), [Weiqiang Wang](), [Jian liu](), [Huan Wang](https://huanwang.tech/), "OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models"

[[Paper]()]

- **2024-11-19:** This repo is released.


![overview](figures/method.png)


> **Abstract:** Omnimodal large language models (OmniLLMs) have attracted increasing research attention of late towards unified audio-video understanding, wherein processing audio-video token sequences creates a significant computational bottleneck, however. Existing token compression methods have yet to accommodate this emerging need of jointly compressing multimodal tokens. To bridge this gap, we present OmniZip, a training-free, audio-guided audio-visual token-compression framework that optimizes multimodal token representation and accelerates inference. Specifically, OmniZip first identifies salient audio tokens, then computes an audio retention score for each time group to capture information density, thereby dynamically guiding video token pruning and preserving cues from audio anchors enhanced by cross-modal similarity. For each time window, OmniZip compresses the video tokens using an interleaved spatio-temporal scheme. Extensive empirical results demonstrate the merits of OmniZip - it achieves 3.42 $\times$ inference speedup and 1.4 $\times$ memory reduction over other top-performing counterparts, while maintaining performance with no training.

## ⚒️ TODO

* [ ] Release Paper 
* [x] Release code 
* [ ] Support more models

## Install
##### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/KD-TAO/OmniZip.git
cd OmniZip
```

##### 2. **Install the inference package:**
```bash
conda create -n omnizip python=3.10 -y
conda activate omnizip
pip install --upgrade pip
bash setup.sh

cd lmms-eval
pip install -e .

# Recommend
# pip install torch==2.6.0 torchvision==0.21.0
pip install flash-attn --no-build-isolation
```
## Quick Start




## Evaluation
#### Set the DyCoke parameters
- We use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit to evaluate our models. It's worth noting that you can specify DyCoke Settings via parameters, such as:
```bash
...
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,dycoke=True,dycoke_l=3,dycoke_p=0.7,dycoke_k=0.7 \
...
```
- Our main baseline model is [LLaVA-OV](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main), if you want to switch between different model frameworks, please change the following parameters:
```bash
...
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen,dycoke=True,dycoke_num_image_per_frame=$YOUR_NUM,image_token_start_index=$YOUR_IDX \
...
```
##### 1. Test on the specified task：
```bash
accelerate launch --num_processes=8 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,dycoke=True \
--tasks $YOUR-TASKS \  # Such as "activitynetqa,video_dc499,perceptiontest_val_mc,videomme_w_subtitle,videomme,nextqa_mc_test..."
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
```
##### 2. **Reproduce the results**：
```bash
bash eval.sh
```
##### 3. **Test on the LLaVA-OV-72B**：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,dycoke=True,device_map=auto \
--tasks $YOUR-TASKS \  # Such as "activitynetqa,video_dc499,perceptiontest_val_mc,videomme_w_subtitle,videomme,nextqa_mc_test..."
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/
```

## 👀 Results on Audio-Video Understanding Task

![overview](figures/table.png)

![overview](figures/teaser.png)
## Acknowledgement

This project is based on [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni). Thanks for their awesome work.

## Contact

If you have any questions, please feel free to contact with me at KD.TAO@outlook.com

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
coming soon
```
