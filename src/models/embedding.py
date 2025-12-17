import os
import numpy as np
from openai import OpenAI

class OpenRouterEmbeddingModel:
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
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
            except Exception:
                pass

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set and could not be read from src/models/openrouter.txt")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """
        Encode a list of texts into embeddings.
        Args:
            texts: List of strings or a single string.
            batch_size: Number of texts to process in one API call.
            show_progress_bar: Unused, kept for compatibility.
        Returns:
            numpy.ndarray of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        # Simple batching
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                # Extract embeddings. response.data is a list of objects with 'embedding' attribute
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch {i}: {e}")
                raise e

        return np.array(all_embeddings)