import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import uuid

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
        self._ensure_collection_exists(
            "COSINE",
            True
        )
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE
        )

    def _ensure_collection_exists(self, distance, auto_create):
        """
        Check collection; create if not exists and allowed by config, else raise.
        """

        try:
            self.client.get_collection(self.collection_name)
        except Exception as e:
            if auto_create:
                vector_size = 1024
                dist = getattr(Distance, distance.upper(), Distance.COSINE)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=dist
                    )
                )
            else:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' does not exist and auto_create_collection=False"
                )

    def _get_embedding_dimension(self):
        """
        Return the embedding model dimension based on known models, with fallback.
        """
        model_name = getattr(self.embeddings, "model", None)

        if model_name:
            if "text-embedding-3" in model_name:
                return 1536
            if "text-embedding-ada-002" in model_name:
                return 1536

        if hasattr(self.embeddings, "dimensions") and self.embeddings.dimensions:
            return self.embeddings.dimensions

        return int(os.getenv("EMBEDDING_DIM", "1536"))

    def add_documents(self, docs, document_uuid=None):
        """
        Add document chunks to the vector database.

        :param docs: List of document chunks to add.
        :param document_uuid: UUID for grouping chunks under a single document.
                              If None, a new UUID is generated.
        :return: List of Qdrant point IDs.
        """
        if document_uuid is None:
            document_uuid = str(uuid.uuid4())

        # Attach document_uuid to each chunk’s metadata
        for doc in docs:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["document_uuid"] = document_uuid

        return self.vectorstore.add_documents(docs)

    def similarity_search(self, query, k=4):
        """
        Search for similar documents using vector similarity.
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)
