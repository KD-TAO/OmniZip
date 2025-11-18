export HF_HOME="YOUR_HF_HOME"

# Or export WRAPPER=None
export WRAPPER=OmniZip
OMNIZIP_RHO_AUDIO=0.3
OMNIZIP_RHO_VIDEO=0.6
OMNIZIP_G=3
OMNIZIP_CONTEXTUAL_RATIO=0.05
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=12347 -m lmms_eval \
    --model qwen2_5_omni \
    --model_args "pretrained=Qwen/Qwen2.5-Omni-7B,attn_implementation=flash_attention_2,max_num_frames=768,OMNIZIP_RHO_AUDIO=${OMNIZIP_RHO_AUDIO},OMNIZIP_RHO_VIDEO=${OMNIZIP_RHO_VIDEO},OMNIZIP_G=${OMNIZIP_G},OMNIZIP_CONTEXTUAL_RATIO=${OMNIZIP_CONTEXTUAL_RATIO}" \
    --tasks videomme \
    --batch_size 1 \
    --output_path ./logs/

# You can use for other benchmarks in lmms-eval