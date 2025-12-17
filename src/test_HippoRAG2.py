import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

# Add project root to path for imports if needed
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from data_classes.data_set import DataSet
import hipporag
from hipporag import HippoRAG

def setup_env():
    """Setup environment variables for OpenRouter/OpenAI compatibility."""
    # Read OpenRouter Key
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
                    api_key = content # Assume raw key?
        except Exception as e:
            print(f"Error reading key file: {e}")
            
    if api_key:
        # HippoRAG/LiteLLM/OpenAI client usually looks for OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = api_key
        print("Set OPENAI_API_KEY from OpenRouter key.")
    else:
        print("Warning: OPENROUTER_API_KEY not found.")

def main():
    print("--- Starting HippoRAG 2 Test (Official Package + OpenRouter Embeddings) ---")
    setup_env()
    
    # 1. Load Data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "HotpotQA_Dev"
    
    # Fallback paths
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
    # Use a small subset to test speed
    docs_subset = dataset.documents[:10] 
    print(f"Loaded {len(dataset.documents)} documents. Using subset of {len(docs_subset)}.")
    
    if not docs_subset:
        print("No documents found.")
        return

    # Prepare docs for HippoRAG (List[str])
    # HippoRAG likely expects just text. Titles might be merged.
    doc_texts = [f"{d.title}\n{d.text}" for d in docs_subset]

    # 2. Initialize HippoRAG
    print("Initializing HippoRAG...")
    # Clean up previous run if needed
    save_dir = "hipporag_test_run"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    rag = HippoRAG(
        llm_model_name="meta-llama/llama-3.3-70b-instruct",
        llm_base_url="https://openrouter.ai/api/v1",
        embedding_model_name="openai/text-embedding-3-small",
        embedding_base_url="https://openrouter.ai/api/v1",
        save_dir=save_dir
    )
    
    # 3. Index Data
    print("Indexing...")
    rag.index(doc_texts)
    
    # 4. Run Test Query
    query = "What is the capital of France?"
    if dataset.qa_pairs:
        query = dataset.qa_pairs[0].question
        print(f"Using Query from dataset: {query}")
    else:
        print(f"Using generic query: {query}")
        
    print(f"Querying: {query}")
    # rag_qa returns a tuple. Based on inspection/docs it might return (solutions, context, metadata, ...)
    try:
        results = rag.rag_qa([query])
        print("\n--- Results ---")
        print(results)
    except Exception as e:
        print(f"Error during query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
