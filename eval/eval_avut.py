"""
Evaluation script for the AVUT Benchmark.
Dataset: https://huggingface.co/datasets/tsinghua-ee/AVUTBenchmark

Usage:
    python eval/eval_avut.py \
        --model-path Qwen/Qwen2.5-Omni-7B \
        --data-path /path/to/AV_Human_data.json \
        --video-dir /path/to/videos \
        --WAPPER-METHOD omnizip \
"""

import json
import os
import sys
import logging
import datetime
import argparse
from collections import defaultdict
from typing import Dict, List

import tqdm
from moviepy import VideoFileClip

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from qwen_omni_utils import process_mm_info


def parse_args():
    parser = argparse.ArgumentParser(description="AVUT Benchmark Evaluation")
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path or HuggingFace ID of the model")
    parser.add_argument('--data-path', type=str, required=True, help="Path to AV_Human_data.json")
    parser.add_argument('--video-dir', type=str, required=True, help="Path to AVUT video directory")
    parser.add_argument('--WAPPER-METHOD', type=str, default="omnizip", help="Compression method: 'omnizip' or 'none'")
    parser.add_argument('--mini-test-num', type=int, default=None, help="If set, only evaluate this many samples")
    parser.add_argument('--OMNIZIP_RHO_AUDIO', type=float, default=0.3, help='Audio token pruning ratio')
    parser.add_argument('--OMNIZIP_RHO_VIDEO', type=float, default=0.6, help='Video token pruning ratio')
    parser.add_argument('--OMNIZIP_G', type=int, default=15, help='Number of tokens merged per anchor (default 15 for AVUT)')
    parser.add_argument('--OMNIZIP_CONTEXTUAL_RATIO', type=float, default=0.05, help='Anchor sampling ratio')
    return parser.parse_known_args()[0]


def check_video_has_audio(video_path):
    try:
        clip = VideoFileClip(video_path)
        has_audio = clip.audio is not None
        clip.close()
        return has_audio
    except Exception:
        return False


def get_candidates_and_answer(sample):
    candidates = []
    for opt in ['A', 'B', 'C', 'D', 'E', 'F']:
        key = f"option_{opt}"
        if key in sample and sample[key] is not None:
            candidates.append(f"{opt}. {sample[key]}")
    correct_answer = sample.get("answer", "").strip().upper()
    return candidates, correct_answer


def extract_answer(resp_text):
    """Extract the predicted option letter from model response."""
    resp_text = resp_text.strip()
    for opt in ["A", "B", "C", "D", "E", "F"]:
        if resp_text.upper().startswith(opt):
            return opt
    if len(resp_text) > 0:
        return resp_text[0].upper()
    return None


def evaluate_sample(sample, model, processor, video_dir, logger):
    """Evaluate a single sample and return the result."""
    video_path = os.path.join(video_dir, sample["video_path"].replace("data/", ""))
    question = sample["question"]
    candidates, correct_answer = get_candidates_and_answer(sample)
    task_type = sample.get("task_type", "Unknown")

    candidates_text = "\n".join(candidates)
    prompt = f"{question}\nOptions:\n{candidates_text}\nAnswer with the option's letter from the given choices directly."

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please analyze the video carefully and select the most appropriate answer from the given options."}],
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
        use_audio = check_video_has_audio(video_path)
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
        model.thinker.nframes = videos[0].shape[0]
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio)
        inputs = inputs.to(model.device).to(model.dtype)

        cont = model.generate(
            **inputs,
            return_audio=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=True,
            temperature=1,
            num_beams=1,
            max_new_tokens=100,
            use_cache=True,
            use_audio_in_video=use_audio,
            thinker_do_sample=False,
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        resp_text = answers[0].strip() if answers and answers[0] else ""
        predicted_answer = extract_answer(resp_text)
        is_correct = (predicted_answer == correct_answer)

        logger.info(f"Processed {sample['video_path']}: GT={correct_answer} Pred={predicted_answer} {'correct' if is_correct else 'wrong'}")
        return {
            "video_id": sample.get("video_id"),
            "QA_id": sample.get("QA_id"),
            "video_path": sample["video_path"],
            "question": question,
            "candidates": candidates,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "task_type": task_type,
            "model_response": resp_text,
        }
    except Exception as e:
        logger.error(f"Error processing {sample['video_path']}: {e}")
        return {
            "video_id": sample.get("video_id"),
            "QA_id": sample.get("QA_id"),
            "video_path": sample["video_path"],
            "question": question,
            "candidates": candidates,
            "correct_answer": correct_answer,
            "predicted_answer": None,
            "is_correct": False,
            "task_type": task_type,
            "model_response": f"Error: {e}",
        }


def calculate_accuracy(results: List[Dict]) -> Dict:
    """Calculate overall and per-task-type accuracy."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])

    task_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        task_stats[r["task_type"]]["total"] += 1
        if r["is_correct"]:
            task_stats[r["task_type"]]["correct"] += 1

    return {
        "overall_accuracy": correct / total if total > 0 else 0,
        "total_correct": correct,
        "total_samples": total,
        "task_type_accuracy": {
            k: {"accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0, **v}
            for k, v in task_stats.items()
        },
    }


def main():
    args = parse_args()

    # Import model class based on method
    if args.WAPPER_METHOD == 'omnizip':
        from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration
    from transformers import Qwen2_5OmniProcessor

    # Setup output directory and logging
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = args.WAPPER_METHOD or "default"
    output_dir = os.path.join("logs", "results_avut", f"{current_time}_{method_tag}")
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'eval_avut.log')),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model from {args.model_path} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )

    if args.WAPPER_METHOD == 'omnizip':
        model.thinker.omnizip_config = {
            "rho_audio": args.OMNIZIP_RHO_AUDIO,
            "rho_video": args.OMNIZIP_RHO_VIDEO,
            "g": args.OMNIZIP_G,
            "contextual_ratio": args.OMNIZIP_CONTEXTUAL_RATIO,
        }
    else:
        model.thinker.omnizip_config = None

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    logger.info("Model and processor loaded successfully")

    # Load benchmark data
    logger.info(f"Loading benchmark data from {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    logger.info(f"Loaded {len(benchmark_data)} samples")

    if args.mini_test_num is not None:
        benchmark_data = benchmark_data[:args.mini_test_num]
        logger.info(f"Mini test: evaluating {len(benchmark_data)} samples")

    # Evaluate
    results = []
    for i, sample in enumerate(tqdm.tqdm(benchmark_data, desc="Evaluating")):
        result = evaluate_sample(sample, model, processor, args.video_dir, logger)
        results.append(result)

        if (i + 1) % 10 == 0:
            with open(os.path.join(output_dir, f"intermediate_results_{i+1}.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy_metrics = calculate_accuracy(results)
    metrics_file = os.path.join(output_dir, "accuracy_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(accuracy_metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("=" * 50)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']:.4f} ({accuracy_metrics['total_correct']}/{accuracy_metrics['total_samples']})")
    logger.info("Accuracy by Task Type:")
    for task_type, stats in accuracy_metrics['task_type_accuracy'].items():
        logger.info(f"  {task_type}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
