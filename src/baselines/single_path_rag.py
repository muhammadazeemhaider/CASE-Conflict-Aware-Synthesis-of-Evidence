"""
Single-Path RAG Baseline

This script mirrors vanilla prompting but augments the prompt with top-k retrieved
passages from a FAISS index. It also logs retrieval metrics (Recall@K, MRR@K)
and downstream QA accuracy.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

from datasets import load_from_disk
from tqdm import tqdm

try:
	from dotenv import load_dotenv
	load_dotenv()
except ImportError:
	pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.base import BaselineRunner
from retrieval.indexing.neural_rag import RAGIndexer

def _compute_retrieval_metrics(
	per_query_ranks: List[int],
	ks: Tuple[int, ...] = (10, 100, 1000),
	) -> Dict[str, Dict[str, float]]:
	"""Compute Recall@K and MRR@K from a list of 1-based ranks (or 0/-1 for miss).
	"""
	n = len(per_query_ranks) or 1
	metrics = {}
	for k in ks:
		hits = 0
		rr_sum = 0.0
		for r in per_query_ranks:
			if r and r > 0 and r <= k:
				hits += 1
				rr_sum += 1.0 / r
		metrics[str(k)] = {
			"recall": hits / n,
			"mrr": rr_sum / n,
		}
	return metrics


def _format_retrieval_block(title: str, metrics: Dict[str, Dict[str, float]]) -> str:
	lines = ["-- Retrieval Scores", ""]
	for k in sorted(metrics.keys(), key=lambda x: int(x)):
		m = metrics[k]
		lines.append(f"Recall@{k}: {m['recall']:.4f}")
		lines.append(f"Mean Reciprocal Rank: {m['mrr']:.4f}")
		lines.append("")
	lines.append("-- Downstream QA task:")
	return "\n".join(lines)


def run_single_path_rag(
	dataset_path: str,
	dataset_name: str = "barexam_qa",
	split: str = "validation",
	index_dir: str = "",
	model: str = "meta-llama/llama-3.3-70b-instruct:free",
	temperature: float = 0.3,
	top_k_context: int = 3,
	retrieval_max_k: int = 1000,
	max_samples: int = None,
	output_dir: str = "",
):
	"""
	Run Single-Path RAG on a dataset and save outputs and metrics.
    """

	# Setup experiment directory
	output_dir = Path(output_dir).resolve()
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	exp_dir = output_dir / f"{dataset_name}_single_path_rag" / timestamp
	exp_dir.mkdir(parents=True, exist_ok=True)

	print(f"Loading QA dataset from {dataset_path}...")
	dataset = load_from_disk(dataset_path)
	data = dataset[split] if isinstance(dataset, dict) else dataset
	if max_samples:
		data = data.select(range(min(max_samples, len(data))))
	print(f"Loaded {len(data)} records\n")

	# Load FAISS retriever
	index_dir = Path(index_dir)
	index_path = os.getenv("FAISS_DIRECTORY_BIN")
	metadata_path = os.getenv("FAISS_DIRECTORY_PKL")
	if not index_path.exists() or not metadata_path.exists():
		raise FileNotFoundError(
			f"Missing FAISS index or metadata at {index_dir}. Expected faiss_index.bin and metadata.pkl"
		)

	retriever = RAGIndexer()
	retriever.load_index(str(index_path), str(metadata_path))

	# Initialize LLM runner
	runner = BaselineRunner(model=model, temperature=temperature, max_tokens=200)

	results = []
	correct = 0
	total_with_gold = 0

	# For retrieval metrics
	gold_ranks: List[int] = []  # 1-based rank when found else 0

	# Bound the retrieval K by index size
	index_size = getattr(getattr(retriever, "index", None), "ntotal", None)
	if index_size is None:
		# Fallback: try metadata length
		index_size = len(retriever.metadata.get("passage_ids", []))
	metrics_k = min(retrieval_max_k, index_size)
	metrics_ks = tuple(sorted(set([10, 100, 1000, metrics_k])))

	for record in tqdm(data, desc="Processing"):
		question = record.get("question", "")
		options = {
			"A": record.get("choice_a", ""),
			"B": record.get("choice_b", ""),
			"C": record.get("choice_c", ""),
			"D": record.get("choice_d", ""),
		}
		gold_answer = record.get("answer")
		gold_idx = record.get("gold_idx")  # string ID
		idx = record.get("idx", None)

		# Retrieve candidates for metrics/context
		k_for_metrics = min(metrics_k, index_size)
		retrieved_all = retriever.search(question, k=max(k_for_metrics, top_k_context))

		# Determine rank of gold passage
		rank = 0
		if gold_idx:
			for item in retrieved_all:
				if str(item.get("passage_id")) == str(gold_idx):
					rank = item.get("rank", 0)
					break
		gold_ranks.append(rank)

		# Build context from top_k_context
		top_k = retrieved_all[:top_k_context]
		context_chunks = [f"[Doc {r['rank']}] {r['passage']}" for r in top_k]
		context = "\n\n---\n\n".join(context_chunks)

		# Query LLM
		result = runner.query(question=question, options=options, context=context, passage_idx=idx)

		# Accuracy bookkeeping
		is_correct = None
		if gold_answer:
			total_with_gold += 1
			is_correct = (result.get("answer") == gold_answer)
			if is_correct:
				correct += 1

		# Save detailed per-example info
		results.append({
			"idx": idx,
			"question": question,
			"model_answer": result.get("answer"),
			"gold_answer": gold_answer,
			"correct": is_correct,
			"gold_idx": gold_idx,
			"retrieved": [
				{
					"passage_id": r.get("passage_id"),
					"split": r.get("split"),
					"rank": r.get("rank"),
					"distance": r.get("distance"),
				}
				for r in retrieved_all
			],
			"topk_context": context,
			"raw_response": result.get("raw_response"),
			"gold_rank": rank if rank else None,
		})

	# Aggregate metrics
	accuracy = (correct / total_with_gold) if total_with_gold else 0.0
	retrieval_metrics = _compute_retrieval_metrics(gold_ranks, ks=(10, 100, 1000))

	# Persist outputs
	with open(exp_dir / "results.jsonl", "w") as f:
		for r in results:
			f.write(json.dumps(r) + "\n")

	stats = {
		"dataset": dataset_name,
		"split": split,
		"model": model,
		"total_samples": len(results),
		"answered_with_gold": total_with_gold,
		"correct": correct,
		"accuracy": round(accuracy, 4),
		"top_k_context": top_k_context,
		"retrieval_metrics": {
			k: {"recall": round(v["recall"], 4), "mrr": round(v["mrr"], 4)}
			for k, v in retrieval_metrics.items()
		},
	}

	with open(exp_dir / "stats.json", "w") as f:
		json.dump(stats, f, indent=2)

	# Also write a dedicated retrieval summary log as JSON (human-friendly + machine-readable)
	retrieval_log = {
		"dataset": dataset_name,
		"split": split,
		"retrieval_scores": retrieval_metrics,  # unrounded floats
		"downstream_qa": {
			"top_k_context": top_k_context,
			"accuracy": accuracy,
			"total_samples": len(results),
		},
		"notes": "Single-Path RAG retrieval evaluation",
	}
	with open(exp_dir / "retrieval_log.json", "w") as f:
		json.dump(retrieval_log, f, indent=2)

	# Console summary in requested style
	print("\n" + "=" * 60)
	print(_format_retrieval_block(dataset_name, retrieval_metrics))
	print(f"\nAccuracy with {{{top_k_context} documents}} retrieved and in context: {accuracy:.4f}")
	print("=" * 60)
	print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run Single-Path RAG baseline")
	parser.add_argument("--dataset-name", default="barexam_qa", help="Dataset name tag for experiment path")
	parser.add_argument("--data-path", default="src/data/downloads/barexam_qa/barexam_qa_qa", help="HuggingFace dataset path for QA")
	parser.add_argument("--split", default="validation", help="Dataset split")
	parser.add_argument("--index-dir", default="src/data/processed/faiss_index_all-mini-llm", help="Directory with faiss_index.bin and metadata.pkl")
	parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free", help="OpenRouter model name")
	parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
	parser.add_argument("--top-k", type=int, default=3, help="Top-K passages to include as context")
	parser.add_argument("--retrieval-max-k", type=int, default=1000, help="Max K for retrieval metrics (bounded by index size)")
	parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of examples")
	parser.add_argument("--output-dir", default="Experiments", help="Output directory base")

	args = parser.parse_args()

	run_single_path_rag(
		dataset_path=args.data_path,
		dataset_name=args.dataset_name,
		split=args.split,
		index_dir=args.index_dir,
		model=args.model,
		temperature=args.temperature,
		top_k_context=args.top_k,
		retrieval_max_k=args.retrieval_max_k,
		max_samples=args.max_samples,
		output_dir=args.output_dir,
	)

