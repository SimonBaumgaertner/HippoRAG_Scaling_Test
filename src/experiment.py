import os
import sys
import time
import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
# Add project root to path for imports if needed
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_classes.data_set import DataSet
from hipporag import HippoRAG

def setup_env():
    """Setup environment variables for OpenRouter/OpenAI compatibility."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        try:
            # Try to read from project root keys
            base_path = Path(__file__).parent.parent
            key_path = base_path / "src" / "models" / "openrouter.txt"
            if key_path.exists():
                content = key_path.read_text().strip()
                if "OPENROUTER_API_KEY=" in content:
                    api_key = content.split("OPENROUTER_API_KEY=")[1].strip()
                else:
                    api_key = content
        except Exception as e:
            print(f"Error reading key file: {e}")
            
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        # print("Set OPENAI_API_KEY from OpenRouter key.")
    else:
        print("Warning: OPENROUTER_API_KEY not found.")

def main():
    print("--- Starting HippoRAG Scaling Experiment ---")
    setup_env()
    
    # Configuration
    SUBSETS = [10, 20, 40]# , 80, 160, 320, 640, 1280]
    RETRIEVAL_QUERY_COUNT = 10
    SAVE_DIR = "hipporag_test_run"
    RESULTS_FILE = "scaling_results.json"
    
    # 1. Load Data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "HotpotQA_Dev"
    
    # Fallback paths logic from test_HippoRAG2.py
    possible_paths = [
        Path("data/HotpotQA_Dev"),
        project_root / "HotpotQA_Dev",
        project_root / "data" / "HotpotQA_Dev"
    ]
    
    final_data_path = None
    for p in possible_paths:
        if p.exists():
            final_data_path = p
            break
            
    if not final_data_path:
        print(f"Error: Data path not found. Checked: {[str(p) for p in possible_paths]}")
        return

    print(f"Loading data from: {final_data_path}")
    dataset = DataSet(final_data_path)
    all_docs = dataset.documents
    print(f"Total documents available: {len(all_docs)}")
    
    # Prepare queries
    all_queries = []
    if dataset.qa_pairs:
        all_queries = [qa.question for qa in dataset.qa_pairs]
    
    # Ensure we have enough queries or cycle
    if not all_queries:
        print("Warning: No queries found in dataset. Using dummy queries.")
        all_queries = ["What is the capital of France?"] * RETRIEVAL_QUERY_COUNT
        
    retrieval_queries = all_queries[:RETRIEVAL_QUERY_COUNT]
    if len(retrieval_queries) < RETRIEVAL_QUERY_COUNT:
        print(f"Warning: Only {len(retrieval_queries)} queries available. Using all of them.")
    
    # 2. Initialize HippoRAG (Clean start)
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
        
    print("Initializing HippoRAG...")
    rag = HippoRAG(
        llm_model_name="meta-llama/llama-3.3-70b-instruct",
        llm_base_url="https://openrouter.ai/api/v1",
        embedding_model_name="openai/text-embedding-3-small",
        embedding_base_url="https://openrouter.ai/api/v1",
        save_dir=SAVE_DIR
    )
    
    dataset_results = []
    
    cumulative_indexing_time = 0.0
    current_doc_count = 0
    
    # 3. Experiment Loop
    for target_count in SUBSETS:
        if target_count > len(all_docs):
            print(f"Stopping: Target count {target_count} exceeds available documents ({len(all_docs)}).")
            break
            
        print(f"\n=== Step: Target {target_count} documents ===")
        
        # Identify new documents to index
        new_docs = all_docs[current_doc_count:target_count]
        new_doc_texts = [f"{d.title}\n{d.text}" for d in new_docs]
        
        # Skip if no new docs (e.g. if SUBSETS has duplicates or logic error)
        if not new_doc_texts:
            print("No new documents to index this step.")
        else:
            print(f"Indexing {len(new_doc_texts)} new documents...")
            start_time = time.time()
            rag.index(new_doc_texts)
            end_time = time.time()
            step_time = end_time - start_time
            cumulative_indexing_time += step_time
            print(f"Indexing Step Time: {step_time:.2f}s")
            
        print(f"Total Indexing Time (Cumulative): {cumulative_indexing_time:.2f}s")
        current_doc_count = target_count
        
        # Retrieval
        print(f"Running Retrieval on {len(retrieval_queries)} queries...")
        retrieval_times = []
        
        for i, query in enumerate(retrieval_queries):
            r_start = time.time()
            try:
                _ = rag.rag_qa([query])
            except Exception as e:
                print(f"Error querying '{query}': {e}")
            r_end = time.time()
            retrieval_times.append(r_end - r_start)
            
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(retrieval_queries)} queries...", end='\r')
                
        avg_retrieval_time = np.mean(retrieval_times) if retrieval_times else 0.0
        print(f"\nAverage Retrieval Time: {avg_retrieval_time:.4f}s")
        
        # Log Result
        result = {
            "document_count": target_count,
            "total_indexing_time_s": cumulative_indexing_time,
            "avg_retrieval_time_s": avg_retrieval_time,
            "queries_run": len(retrieval_times)
        }
        dataset_results.append(result)
        
        # Save intermediate results
        with open(RESULTS_FILE, 'w') as f:
            json.dump(dataset_results, f, indent=2)
            
    print(f"\nExperiment Completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
