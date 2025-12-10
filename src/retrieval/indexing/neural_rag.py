"""
Simple FAISS-based indexing for Bar Exam QA dataset using HuggingFace embeddings.
"""

import numpy as np
import faiss
from pathlib import Path
from datasets import load_from_disk 
from sentence_transformers import SentenceTransformer
import pickle

class RAGIndexer:
    """
    A simple indexer for Bar Exam QA passages using FAISS and HuggingFace embeddings.
    
    Uses 'all-MiniLM-L6-v2' model:
    - 384 dimensions (lightweight)
    - Fast and efficient
    - Good performance on semantic search tasks
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the indexer with embedding model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.passages = None
        self.metadata = None  # To store passage IDs and other info
    
    def build_index(self, data_path, output_dir):
        """
        Build FAISS index from Bar Exam passages dataset.
        
        Args:
            data_path: Path to the barexam_qa_passages dataset
            output_dir: Directory to save the index 
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the dataset
        print(f"Loading dataset from {data_path}...")
        dataset = load_from_disk(data_path)
        
        # Combine all splits (train, validation, test)
        all_passages = []
        all_passage_ids = []
        all_splits = []
        
        print(f"Available splits: {dataset.keys()}")
        
        for split_name in dataset.keys():
            passages_data = dataset[split_name]
            split_size = len(passages_data)
            
            print(f"  - {split_name}: {split_size} passages")
            
            all_passages.extend(passages_data["text"])
            all_passage_ids.extend(passages_data["idx"])
            all_splits.extend([split_name] * split_size)
        
        total_passages = len(all_passages)
        print(f"\nTotal passages from all splits: {total_passages}")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(
            all_passages,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        print("Building FAISS index...")
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.passages = all_passages
        self.metadata = {
            "passage_ids": all_passage_ids,
            "passage_texts": all_passages,
            "splits": all_splits  # Track which split each passage comes from
        }
        
        # Save index and metadata
        index_path = output_dir / "faiss_index.bin"
        metadata_path = output_dir / "metadata.pkl"
        
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Print split distribution
        split_counts = {}
        for split in all_splits:
            split_counts[split] = split_counts.get(split, 0) + 1
        
        print(f"\nIndex built successfully!")
        print(f"  - Index saved to: {index_path}")
        print(f"  - Metadata saved to: {metadata_path}")
        print(f"  - Total passages indexed: {len(all_passages)}")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Split distribution:")
        for split, count in split_counts.items():
            print(f"      {split}: {count}")
    
    # def load_index(self, index_path, metadata_path):
    #     """Load pre-built FAISS index and metadata."""
    #     print(f"Loading index from {index_path}...")
    #     self.index = faiss.read_index(str(index_path))
        
    #     with open(metadata_path, "rb") as f:
    #         self.metadata = pickle.load(f)
        
    #     self.passages = self.metadata["passage_texts"]
    #     print(f"Index loaded successfully. Total passages: {len(self.passages)}")
    
    def load_index(self, index_path, metadata_path):
        """Load pre-built FAISS index and metadata."""
        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        # Extract passages from id_to_text mapping
        if 'id_to_text' in self.metadata:
            # Convert id_to_text dict to list
            num_passages = len(self.metadata['id_to_text'])
            self.passages = [self.metadata['id_to_text'][i] for i in range(num_passages)]
        elif 'passage_texts' in self.metadata:
            # Fallback for old format
            self.passages = self.metadata["passage_texts"]
        else:
            raise KeyError("Metadata must contain either 'id_to_text' or 'passage_texts'")
        
        print(f"Index loaded successfully. Total passages: {len(self.passages)}")

    def search(self, query, k=5):
        """
        Search for similar passages using a query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of dicts with passage info and metadata
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results using the actual metadata structure
        results = []
        for i, idx in enumerate(indices[0]):
            # Access text from id_to_text mapping
            passage_text = self.metadata.get("id_to_text", {}).get(int(idx), "")
            
            # Access other metadata fields
            passage_idx = self.metadata.get("id_to_idx", {}).get(int(idx))
            case_id = self.metadata.get("id_to_case_id", {}).get(int(idx))
            opinion_id = self.metadata.get("id_to_opinion_id", {}).get(int(idx))
            abs_para_id = self.metadata.get("id_to_absolute_paragraph_id", {}).get(int(idx))
            rel_para_id = self.metadata.get("id_to_relative_paragraph_id", {}).get(int(idx))
            faiss_id = self.metadata.get("id_to_faiss_id", {}).get(int(idx))
            
            distance = distances[0][i]
            
            results.append({
                "passage": passage_text,
                "idx": passage_idx,
                "case_id": case_id,
                "opinion_id": opinion_id,
                "absolute_paragraph_id": abs_para_id,
                "relative_paragraph_id": rel_para_id,
                "faiss_id": faiss_id,
                "distance": float(distance),
                "rank": i + 1
            })
        
        return results

    # def search(self, query, k=5):
    #     """
    #     Search for similar passages using a query.
        
    #     Args:
    #         query: Query text
    #         k: Number of results to return
            
    #     Returns:
    #         List of (passage, passage_id, split, distance) tuples
    #     """
    #     if self.index is None:
    #         raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
    #     # Encode query
    #     query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        
    #     # Search
    #     distances, indices = self.index.search(query_embedding, k)
        
    #     # Prepare results
    #     results = []
    #     for i, idx in enumerate(indices[0]):
    #         passage_text = self.passages[idx]
    #         passage_id = self.metadata["passage_ids"][idx]
    #         split = self.metadata.get("splits", ["unknown"] * len(self.passages))[idx]
    #         distance = distances[0][i]
            
    #         results.append({
    #             "passage": passage_text,
    #             "passage_id": passage_id,
    #             "split": split,
    #             "distance": float(distance),
    #             "rank": i + 1
    #         })
        
    #     return results

def main():
    """Example usage of the RAGIndexer."""
    # Paths
    base_dataset = 'barexam_qa/barexam_qa_passages'
    data_path = f"src/data/downloads/{base_dataset}"
    index_output_dir = f"src/data/processed/faiss_index/{base_dataset}"
    
    # Create indexer
    indexer = RAGIndexer()
    
    # Build and save index
    indexer.build_index(data_path, index_output_dir)
    
    # Example query
    test_query = "What is contract law?"
    print(f"\nSearching for: '{test_query}'")
    results = indexer.search(test_query, k=3)
    
    print("\nTop results:")
    for result in results:
        print(f"  Rank {result['rank']}: {result['passage'][:100]}...")
        print(f"    ID: {result['passage_id']}, Split: {result['split']}, Distance: {result['distance']:.4f}\n")


if __name__ == "__main__":
    main()
