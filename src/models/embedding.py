import os
import time
import numpy as np
from typing import List, Optional
from openai import OpenAI
from hipporag.embedding_model.base import BaseEmbeddingModel
from hipporag.utils.config_utils import BaseConfig

class OpenRouterEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, global_config: Optional[BaseConfig] = None, model_name: str = "openai/text-embedding-3-small"):
        # Initialize parent
        super().__init__(global_config=global_config)
        
        # Determine model name: explicit arg > config > default
        if self.global_config and self.global_config.embedding_model_name:
            self.model_name = self.global_config.embedding_model_name
        else:
            self.model_name = model_name

        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Try to read from src/models/openrouter.txt
            try:
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Root of project from src/models/embedding.py
                key_path = os.path.join(base_path, "src", "models", "openrouter.txt")
                if os.path.exists(key_path):
                    with open(key_path, "r") as f:
                        for line in f:
                            if line.startswith("OPENROUTER_API_KEY="):
                                self.api_key = line.strip().split("=", 1)[1]
                                break
                            elif not line.startswith("#") and "openrouter" not in line.lower() and len(line.strip()) > 20: 
                                # Fallback: assume the whole line is the key if it looks like one
                                self.api_key = line.strip()
            except Exception:
                pass

        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not found. Helper might fail.")
        
        # Initialize client
        # Use OpenRouter base URL
        base_url = "https://openrouter.ai/api/v1"
        if self.global_config and self.global_config.embedding_base_url:
            base_url = self.global_config.embedding_base_url
            
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )

    def encode(self, texts: List[str]):
        # OpenAI/OpenRouter specific: replace newlines
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        
        # Retry logic
        max_retries = 5
        backoff = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                
                if not response.data:
                    raise ValueError("Response data is empty or None")
                    
                results = np.array([v.embedding for v in response.data])
                return results
                
            except Exception as e:
                print(f"Error generating embeddings (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise e

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        batch_size = kwargs.get("batch_size", 32)
        if self.global_config and self.global_config.embedding_batch_size:
            batch_size = self.global_config.embedding_batch_size
            
        all_embeddings = []
        
        # Simple batching
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encode(batch)
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return np.array([])
            
        return np.concatenate(all_embeddings)