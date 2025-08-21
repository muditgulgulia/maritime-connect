import os
from fastapi import APIRouter, UploadFile, Form, HTTPException
from app.services.document_loader import DocumentLoader
from app.services.text_splitter import TextSplitter
from app.services.embedding_model import EmbeddingModel
from app.services.qdrant_vectordb import QdrantVectorDB

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Initialize embedding and vector DB service
embedding_model = EmbeddingModel().get()

@router.post("/new")
async def ingest_manual(uuid: str = Form(...), file: UploadFile = Form(...)):
    """
    Ingest a manual PDF into Qdrant.
    - User sends `uuid` and `file` (PDF).
    - No database is queried.
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uuid}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Initialize pipeline services
        loader = DocumentLoader()
        splitter = TextSplitter()
        vector_db = QdrantVectorDB(collection_name=os.getenv("QDRANT_COLLECTION", "maritime"), embeddings=embedding_model)

        # Load → Split
        docs = loader.load(temp_path)
        chunks = splitter.split(docs)

        # Add chunks to Qdrant using provided UUID
        vector_db.add_documents(chunks, document_uuid=uuid)

        return {
            "uuid": uuid,
            "chunks_added": len(chunks),
            "status": "ingested"
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Manual ingestion failed.")
