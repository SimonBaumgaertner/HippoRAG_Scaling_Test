import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from embedding import OpenRouterEmbeddingModel
from llm import OpenRouterLLM

def verify_embedding():
    print("--- Verifying Embedding Model ---")
    try:
        model = OpenRouterEmbeddingModel()
        text = ["Hello world", "HippoRAG is cool"]
        embeddings = model.encode(text)
        print(f"Encoded {len(text)} texts.")
        print(f"Embeddings shape: {embeddings.shape}")
        assert embeddings.shape[0] == 2
        # Qwen 0.5B embedding dim should be 1024 (likely) or 768.
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print("Embedding test PASSED.")
    except Exception as e:
        print(f"Embedding test FAILED: {e}")
        import traceback
        traceback.print_exc()

def verify_llm():
    print("\n--- Verifying LLM Model ---")
    
    try:
        llm = OpenRouterLLM()
        prompt = "What is the capital of France?"
        print(f"Prompt: {prompt}")
        response = llm.generate(prompt)
        print(f"Response: {response}")
        if response and len(response) > 0:
            print("LLM test PASSED.")
        else:
            print("LLM test FAILED (Empty response).")
    except Exception as e:
        print(f"LLM test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_embedding()
    verify_llm()
