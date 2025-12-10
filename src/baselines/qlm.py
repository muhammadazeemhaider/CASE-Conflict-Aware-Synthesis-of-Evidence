"""
QLM + BaselineRunner pipeline for BarExamQA validation set.
Runs retrieval for k = 3, 5, and 10.
Implements Query Likelihood Model retrieval with Dirichlet smoothing.
"""

import json
import math
import time
from collections import Counter
from datasets import load_dataset, load_from_disk
from .base import BaselineRunner
from dotenv import load_dotenv
import os

# === Load environment variables ===
load_dotenv()

# === Configuration ===
DATASET_NAME = "reglab/barexam_qa"
DATA_SPLIT = "validation"
K_VALUES = [3, 5, 10] 
MU = 2000 
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"
OUTPUT_DIR = "Experiments//qlm" # Folder where results will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load dataset ===
# Load local datasets if available
qa_path = os.getenv("DATASET_PATH_QA", "")
passages_path = os.getenv("DATASET_PATH_PASSAGES", "")

if os.path.exists(qa_path) and os.path.exists(passages_path):
    print(f"Loading local datasets from {qa_path} and {passages_path}")
    dataset = load_from_disk(qa_path)
    dataset = dataset["validation"] if "validation" in dataset else dataset["train"]
    passages = load_from_disk(passages_path)
    passages = passages["train"]
else:
    print("Local dataset not found. Falling back to Hugging Face.")
    dataset = load_dataset(DATASET_NAME, name="qa")[DATA_SPLIT]
    passages = load_dataset(DATASET_NAME, name="passages")["train"]

# Build corpus
corpus = {str(i): p["text"] for i, p in enumerate(passages)}

# Tokenize helper
def tokenize(text):
    return text.lower().split()

# === Precompute stats ===
print("Building collection statistics...")
collection_tokens = []
doc_freqs = {}
doc_lengths = {}

for pid, text in corpus.items():
    tokens = tokenize(text)
    freqs = Counter(tokens)
    doc_freqs[pid] = freqs
    doc_lengths[pid] = len(tokens)
    collection_tokens.extend(tokens)

collection_len = len(collection_tokens)
collection_freq = Counter(collection_tokens)
collection_prob = {t: c / collection_len for t, c in collection_freq.items()}

def qlm_score(query, doc_id):
    """Compute log P(Q|D) using Dirichlet smoothing."""
    tokens = tokenize(query)
    doc_f = doc_freqs[doc_id]
    doc_len = doc_lengths[doc_id]
    score = 0.0
    for t in tokens:
        f_td = doc_f.get(t, 0)
        p_t_C = collection_prob.get(t, 1e-12)
        p_t_D = (f_td + MU * p_t_C) / (doc_len + MU)
        score += math.log(p_t_D + 1e-12)
    return score

def retrieve(query, top_k):
    """Retrieve top-k documents using QLM scoring."""
    start = time.time()
    scores = {pid: qlm_score(query, pid) for pid in corpus}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    end = time.time()
    return ranked, (end - start)

# === Initialize LLM baseline ===
runner = BaselineRunner(model=MODEL_NAME, temperature=0.3, max_tokens=150)

# === Run for each K value ===
for TOP_K in K_VALUES:
    print(f"\n=== Running QLM retrieval with TOP_K = {TOP_K} ===")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"barexamqa_qlm_rag_results_k{TOP_K}.json")

    results = []
    timings = []

    for i, record in enumerate(dataset):
        question = record.get("question", "")
        options = {
            "A": record.get("choice_a", ""),
            "B": record.get("choice_b", ""),
            "C": record.get("choice_c", ""),
            "D": record.get("choice_d", ""),
        }

        # --- Retrieve top-k passages ---
        retrieved, retrieval_time = retrieve(question, top_k=TOP_K)
        top_docs = [corpus[pid] for pid, _ in retrieved]
        context = "\n\n".join(top_docs)

        start_llm = time.time()
        result = runner.query(
            question=question,
            options=options,
            context=context,
            passage_idx=record["idx"]
        )
        llm_time = time.time() - start_llm

        # --- Combine metadata ---
        result.update({
            "question": question,
            "choices": options,
            "context_passages": top_docs,
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": retrieval_time + llm_time,
            "k_value": TOP_K
        })
        results.append(result)
        timings.append(result["total_time"])

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} for k={TOP_K}")
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # === Save final results for this K ===
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {OUTPUT_FILE}")
    print(f"Average response time (k={TOP_K}): {sum(timings)/len(timings):.2f}s")
