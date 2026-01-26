import json
import os
import logging
import tqdm
import sys
from moviepy import VideoFileClip
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from typing import Dict, List
from transformers import Qwen2_5OmniProcessor
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--WAPPER-METHOD', type=str, default="omnizip", help='WAPPER-METHOD parameter')
parser.add_argument('--mini-test-num', type=int, default=None, help="If set, only test this number of samples (mini test).")
parser.add_argument('--OMNIZIP_RHO_AUDIO', type=float, default=0.3, help='Merging ratio for audio')
parser.add_argument('--OMNIZIP_RHO_VIDEO', type=float, default=0.6, help='Merging ratio for video')
parser.add_argument('--OMNIZIP_G', type=int, default=3, help='G parameter')
parser.add_argument('--OMNIZIP_CONTEXTUAL_RATIO', type=float, default=0.05, help='Contextual ratio')
args, unknown = parser.parse_known_args()
WAPPER_METHOD = args.WAPPER_METHOD
MINI_TEST_NUM = args.mini_test_num
OMNIZIP_RHO_AUDIO = args.OMNIZIP_RHO_AUDIO
OMNIZIP_RHO_VIDEO = args.OMNIZIP_RHO_VIDEO
OMNIZIP_G = args.OMNIZIP_G
OMNIZIP_CONTEXTUAL_RATIO = args.OMNIZIP_CONTEXTUAL_RATIO

if WAPPER_METHOD == 'omnizip':
    print('Using omni method')
    from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
else:
    print("Not found WAPPER-METHOD")
    from transformers import Qwen2_5OmniForConditionalGeneration

from qwen_omni_utils import process_mm_info


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eval_worldsense.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = os.path.join("Data", "WorldSense", "worldsense_qa.json")
VIDEO_DIR = os.path.join("Data", "WorldSense", "videos")
MODEL_PATH = "huggingface/Qwen2.5-Omni-3B"

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_prefix = f"{current_time}_{WAPPER_METHOD}" if WAPPER_METHOD else f"{current_time}_default"
OUTPUT_DIR = os.path.join("logs/results_worldsense", log_prefix)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model and processor
logger.info("Loading model and processor...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

omnizip_config = {
    "rho_audio": OMNIZIP_RHO_AUDIO,
    "rho_video": OMNIZIP_RHO_VIDEO,
    "g": OMNIZIP_G,
    "contextual_ratio": OMNIZIP_CONTEXTUAL_RATIO,
}

if WAPPER_METHOD == 'omnizip':  
    model.thinker.omnizip_config = omnizip_config
else:
    model.thinker.omnizip_config = None


processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
logger.info("Model and processor loaded successfully")

# Load benchmark data
logger.info(f"Loading benchmark data from {DATA_PATH}")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# Reformat json into a flat list of samples (for all tasks)
benchmark_data = []
for vid, v in qa_data.items():
    for key in v:
        if key.startswith("task"):
            task_obj = v[key]
            sample = {
                "video": v["video_id"] + ".mp4",  # videos are in .mp4 files
                "video_id": v["video_id"],
                "video_caption": v.get("video_caption", ""),
                "problem_type": task_obj.get("task_type", key),
                "data_type": task_obj.get("task_domain", ""),
                "domain": v.get("domain", ""),  # <--- 新增domain字段
                "question": task_obj["question"],
                "answer": task_obj["answer"],
                "candidates": task_obj["candidates"]
            }
            benchmark_data.append(sample)

logger.info(f"Loaded {len(benchmark_data)} samples")

if MINI_TEST_NUM is not None:
    logger.info(f"Mini test is ON: will only evaluate {MINI_TEST_NUM} samples")
    benchmark_data = benchmark_data[:MINI_TEST_NUM]

def _check_if_video_has_audio(video_path):
    try:
        clip = VideoFileClip(video_path)
        return clip.audio is not None
    except Exception as e:
        logger.error(f"Failed to load video/audio {video_path}: {e}")
        return False

def evaluate_sample(sample: Dict) -> Dict:
    """Evaluate a single sample and return the result"""
    video_path = os.path.join(VIDEO_DIR, sample["video"])
    question = sample["question"]
    candidates = sample["candidates"]
    correct_answer = sample["answer"]
    
    # Format candidates as text
    candidates_text = "\n".join(candidates)
    prompt = f"{question}\nOptions:\n{candidates_text}\nAnswer with the option's letter from the given choices directly."
    print(prompt)
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please analyze the video carefully and select the most appropriate answer from the given options."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    try:
        # Preparation for inference
        USE_AUDIO_IN_VIDEO = _check_if_video_has_audio(video_path)
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        num_input_frames = videos[0].shape[0]
        model.thinker.nframes = num_input_frames
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)
        
        cont = model.generate(
                    **inputs,
                    return_audio=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=1,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=100,
                    use_cache=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    thinker_do_sample=False,
                )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        predicted_answer = None
        resp_text = answers[0].strip() if answers and answers[0] else ""
        for opt in ["A", "B", "C", "D", "E", "F"]:
            if resp_text.upper().strip().startswith(opt):
                predicted_answer = opt
                break
        if predicted_answer is None and len(resp_text) > 0:
            predicted_answer = resp_text[0].upper()
        is_correct = (predicted_answer == correct_answer)

        result = {
            "video": sample["video"],
            "video_id": sample.get("video_id", ""),
            "question": question,
            "candidates": candidates,
            "video_caption": sample.get("video_caption", ""),
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "problem_type": sample["problem_type"],
            "data_type": sample["data_type"],
            "domain": sample["domain"],
            "model_response": resp_text
        }
        
        logger.info(f"Processed {sample['video']}: {correct_answer} -> {predicted_answer} ({'✓' if is_correct else '✗'})")
        return result
        
    except Exception as e:
        logger.error(f"Error processing {sample['video']}: {str(e)}")
        return {
            "video": sample["video"],
            "video_id": sample.get("video_id", ""),
            "question": question,
            "candidates": candidates,
            "video_caption": sample.get("video_caption", ""),
            "correct_answer": correct_answer,
            "predicted_answer": None,
            "is_correct": False,
            "problem_type": sample["problem_type"],
            "data_type": sample["data_type"],
            "domain": sample.get("domain", ""),
            "model_response": f"Error: {str(e)}"
        }

def calculate_accuracy(results: List[Dict]) -> Dict:
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r["is_correct"])
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        domain = result.get("domain", "")
        domain_stats[domain]["total"] += 1
        if result["is_correct"]:
            domain_stats[domain]["correct"] += 1

    domain_accuracy = {}
    for domain, stats in domain_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        domain_accuracy[domain] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"]
        }

    problem_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        problem_type = result["problem_type"]
        problem_type_stats[problem_type]["total"] += 1
        if result["is_correct"]:
            problem_type_stats[problem_type]["correct"] += 1
    problem_type_accuracy = {}
    for problem_type, stats in problem_type_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        problem_type_accuracy[problem_type] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"]
        }

    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": correct_predictions,
        "total_samples": total_samples,
        "domain_accuracy": domain_accuracy,
        "problem_type_accuracy": problem_type_accuracy,
    }

logger.info("Starting evaluation...")
results = []
test_loop_range = benchmark_data if MINI_TEST_NUM is None else benchmark_data[:MINI_TEST_NUM]
for i, sample in enumerate(tqdm.tqdm(test_loop_range, desc="Evaluating samples")):
    logger.info(f"Processing sample {i+1}/{len(test_loop_range)}: {sample['video']}")
    result = evaluate_sample(sample)
    results.append(result)

    if MINI_TEST_NUM is not None and (i + 1) >= MINI_TEST_NUM:
        logger.info(f"Mini test: evaluated {MINI_TEST_NUM} samples, now stopping.")
        break

results_file = os.path.join(OUTPUT_DIR, "evaluation_results.json")
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
logger.info(f"Saved final results to {results_file}")

# Calculate and save accuracy metrics
accuracy_metrics = calculate_accuracy(results)
metrics_file = os.path.join(OUTPUT_DIR, "accuracy_metrics.json")
with open(metrics_file, 'w', encoding='utf-8') as f:
    json.dump(accuracy_metrics, f, indent=2, ensure_ascii=False)

# Log final results
logger.info("=" * 50)
logger.info("EVALUATION COMPLETED")
logger.info("=" * 50)
logger.info(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']:.4f} ({accuracy_metrics['total_correct']}/{accuracy_metrics['total_samples']})")
logger.info("\nAccuracy by Domain:")
for domain, stats in accuracy_metrics['domain_accuracy'].items():
    logger.info(f"  {domain}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
logger.info("\nAccuracy by Problem Type:")
for problem_type, stats in accuracy_metrics['problem_type_accuracy'].items():
    logger.info(f"  {problem_type}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

logger.info(f"\nDetailed results saved to: {results_file}")
logger.info(f"Accuracy metrics saved to: {metrics_file}")