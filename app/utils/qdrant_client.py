import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

class QdrantVectorDB:
    """
    Manages storage and retrieval of vectors in Qdrant. Automatically creates collection if it does not exist.
    """

    def __init__(self, collection_name, embeddings):
        """
        Initialize the Qdrant vector database for a specific collection, create if missing.
        """
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=300
        )
        
        # self.client.create_collection(
        #     collection_name=collection_name,
        #     vectors_config=qdrant_models.VectorParams(
        #         size=1024,
        #         distance=qdrant_models.Distance.COSINE
        #     )
        # )

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE
        )

    def similarity_search(self, query, k=4):
        """
        Search for similar documents using vector similarity.
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)
