"""
BM25 + BaselineRunner pipeline for BarExamQA validation set.
No changes to base.py are required.
"""

import json
from pyserini.search.lucene import LuceneSearcher as SimpleSearcher
from .base import BaselineRunner
from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv() 

# === Configuration ===
INDEX_PATH = ""
TOP_K = 5
OUTPUT_FILE = ""
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"

# === Load dataset ===
dataset = load_dataset("reglab/barexam_qa", name="qa")

# === Initialize searcher and runner ===
searcher = SimpleSearcher(INDEX_PATH)
runner = BaselineRunner(model=MODEL_NAME, temperature=0.3, max_tokens=150)

results = []

for i, record in enumerate(dataset):
    question = record.get("question", "")
    options = {
        "A": record.get("choice_a", ""),
        "B": record.get("choice_b", ""),
        "C": record.get("choice_c", ""),
        "D": record.get("choice_d", ""),
    }

    # --- Retrieve top-k passages from BM25 ---
    hits = searcher.search(question, k=TOP_K)
    retrieved = [searcher.doc(hit.docid).raw() for hit in hits]
    context = "\n\n".join(retrieved)

    # --- Query LLM with context ---
    try:
        result = runner.query(
            question=question,
            options=options,
            context=context,
            passage_idx=record["idx"]
        )

        # Check if the model flagged or returned an error
        if result.get("error") and "requires moderation" in result["error"]:
            print(f"⚠️ Skipped question {i} due to moderation flag.")
            result["answer"] = None
            result["raw_response"] = "FLAGGED_BY_MODERATION"

    except Exception as e:
        print(f"❌ Error on question {i}: {e}")
        result = {
            "answer": None,
            "raw_response": str(e),
            "passage_idx": i,
            "error": str(e)
        }

    # --- Merge with metadata for tracking ---
    result.update({
        "question": question,
        "choices": options,
        "context_passages": retrieved
    })
    results.append(result)

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(dataset)} questions")
        print(f"Saving intermediate results to {OUTPUT_FILE}")
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Handle potential errors during saving, but keep running
            print(f"Error during intermediate save: {e}")

# === Save results ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} results to {OUTPUT_FILE}")
