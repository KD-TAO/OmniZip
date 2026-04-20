import argparse

import torch
from transformers import Qwen3OmniMoeProcessor, TextStreamer
from qwen_omni_utils import process_mm_info


parser = argparse.ArgumentParser(description="Qwen3-Omni Demo Parameters")
parser.add_argument('--model_path', type=str,
                    default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    help='HuggingFace model id or local path to Qwen3-Omni weights')
parser.add_argument('--video', type=str, default="assets/example.mp4", help='Path to input video file')
parser.add_argument('--describe', type=str, default="Describe this video.", help='Instruction text for the model')
parser.add_argument('--omnizip', action='store_true', help='Whether to enable omnizip')
parser.add_argument('--rho_audio', type=float, default=0.4, help='Merging ratio for audio')
parser.add_argument('--rho_video', type=float, default=0.7, help='Merging ratio for video')
parser.add_argument('--g', type=int, default=3)
parser.add_argument('--contextual_ratio', type=float, default=0.05, help='Contextual ratio')
args = parser.parse_args()

if args.omnizip:
    print("Using OmniZip for Inference")
    omnizip_config = {
        "rho_audio": args.rho_audio,
        "rho_video": args.rho_video,
        "g": args.g,
        "contextual_ratio": args.contextual_ratio,
    }
    from omnizip.qwen3_omni.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
else:
    from transformers import Qwen3OmniMoeForConditionalGeneration

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

if args.omnizip:
    model.thinker.omnizip_config = omnizip_config

processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                "capable of perceiving auditory and visual inputs, as well as generating text and speech."
            )}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": args.video},
            {"type": "text", "text": args.describe},
        ],
    },
]

USE_AUDIO_IN_VIDEO = True

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO,
)
inputs = inputs.to(model.device).to(model.dtype)

print('Generating...')
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
text_ids = model.generate(
    **inputs,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
    return_audio=False,
    temperature=1,
    streamer=streamer,
)
