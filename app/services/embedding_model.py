import os
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

class EmbeddingModel(Embeddings):
    """
    Wrapper around DashScope embeddings that:
    - Works with LangChain/Qdrant
    - Calls the official OpenAI client (DashScope-compatible) for real embeddings
    - Avoids Qdrant's "dummy_text" auto-check error
    """

    def __init__(self, model_name: str = "text-embedding-v3"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

        self.model = OpenAIEmbeddings(
            model=model_name,
            base_url=self.base_url,
            openai_api_key=self.api_key,
        )

        # Direct OpenAI client (for stable embedding calls)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get(self):
        """
        Return the embedding model.
        """
        return self

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        """Safe embedding for multiple docs."""
        
        if docs == ["dummy_text"]:
            return [[0.0] * 1024]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=docs,
            encoding_format="float"
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Safe embedding for a single query string."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding