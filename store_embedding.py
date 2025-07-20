from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_handler import QdrantStore
from typing import List, Optional
from embedd import EmbedDocuments
import os 

app = FastAPI()
class StoreEmbeddings(BaseModel):
    pdf_path : List[str] = []
@app.post("/insert_texts")
async def insert_pdf_texts(req:StoreEmbeddings, collection_name: str = "neurosurgery", url: str = "http://localhost:6333"):
    embedd_docs = EmbedDocuments(collection_name=collection_name, url=url)
    for pdf in req.pdf_path:
        if not os.path.exists(pdf):
            raise HTTPException(status_code=400, detail=f"File not found: {pdf}")
        embedd_docs.embed_and_store(pdf)
    return {"message": "PDF texts inserted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)