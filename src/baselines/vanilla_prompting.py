"""
Vanilla Prompting Baseline - Simple script to run QA on any multiple-choice dataset
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
from tqdm import tqdm
from datasets import load_dataset

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.base import BaselineRunner


def run_vanilla_prompting(
    dataset_path,
    dataset_name="dataset",
    split="validation",
    model="meta-llama/llama-3.3-70b-instruct:free",
    temperature=0.3,
    max_samples=None,
    output_dir="../../Experiments",
):
    """
    Run vanilla prompting on a dataset and save results.
    
    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name for experiment (e.g., 'barexam_qa')
        split: Dataset split ('train', 'validation', 'test')
        model: OpenRouter model name
        temperature: Model temperature
        max_samples: Optional limit on samples
        output_dir: Where to save results
    """
    
    # Setup output directory
    output_dir = Path(output_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"{dataset_name}_vanilla_prompting" / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    # dataset = load_from_disk(dataset_path)
    dataset = load_dataset("reglab/barexam_qa", name="qa", trust_remote_code=True)
    dataset = dataset["validation"]
    if isinstance(dataset, dict):
        data = dataset[split]
    else:
        data = dataset
    
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    print(f"Loaded {len(data)} records\n")
    
    # Initialize runner
    runner = BaselineRunner(model=model, temperature=temperature, max_tokens=100)
    
    # Run inference
    results = []
    correct = 0
    
    for record in tqdm(data, desc="Processing"):
        try:
            question = record.get("question", "")
            options = {
                "A": record.get("choice_a", ""),
                "B": record.get("choice_b", ""),
                "C": record.get("choice_c", ""),
                "D": record.get("choice_d", ""),
            }
            gold_answer = record.get("answer")
            gold_passage = record.get("gold_passage")
            # print(gold_passage)
            idx = record.get("idx", 0)
            
            # Query model
            result = runner.query(question=question, options=options, gold_passage=gold_passage, passage_idx=idx)
            
            # Check if correct
            is_correct = result["answer"] == gold_answer if gold_answer else None
            if is_correct:
                correct += 1
            
            # Save result
            results.append({
                "idx": idx,
                "question": question,
                "model_answer": result["answer"],
                "gold_passage": gold_passage,
                "options": options,
                "gold_answer": gold_answer,
                "correct": is_correct,
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({"idx": record.get("idx"), "error": str(e)})
    
    # Compute accuracy
    total_with_gold = sum(1 for r in results if r.get("gold_answer"))
    accuracy = (correct / total_with_gold * 100) if total_with_gold > 0 else 0
    
    # Save results
    with open(exp_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Save stats
    stats = {
        "dataset": dataset_name,
        "split": split,
        "model": model,
        "total_samples": len(results),
        "correct": correct,
        "accuracy": round(accuracy, 2),
    }
    
    with open(exp_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {exp_dir}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total_with_gold})")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run vanilla prompting baseline")
    parser.add_argument("--dataset-name", default="barexam_qa", help="Dataset name")
    parser.add_argument("--data-path", default="data/downloads/barexam_qa/barexam_qa_qa", help="Dataset path")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument("--output-dir", default="Experiments", help="Output directory")
    
    args = parser.parse_args()
    
    run_vanilla_prompting(
        dataset_path=args.data_path,
        dataset_name=args.dataset_name,
        split=args.split,
        model=args.model,
        temperature=args.temperature,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )