import os
from openai import OpenAI

class OpenRouterLLM:
    def __init__(self, model_name: str = "meta-llama/llama-3.3-70b-instruct"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            # Try to read from src/models/openrouter.txt
            try:
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Root of project from src/models/llm.py
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

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        Args:
            prompt: The input prompt string.
        Returns:
            The generated response string.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            raise e
