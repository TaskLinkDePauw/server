# pipeline/supplier_pdf_ingestion.py
import os
import openai
import re
from typing import List
from sqlalchemy.orm import Session
from .chunking_utils import read_and_chunk_pdf_adaptive
from .embedding_utils import batch_embed_texts
import repository, schemas

def detect_roles_from_text(text_snippet: str, openai_api_key: str, num_roles=3) -> List[str]:
    """
    Use OpenAI to guess up to 'num_roles' possible roles from the text.
    E.g. ["plumber", "electrician"].
    """
    if not openai_api_key:
        return []

    prompt = f"""
    The following text is from a resume or profile.
    Identify up to {num_roles} distinct professional roles or services the candidate can offer.
    Return them as a comma-separated list, all lowercase, short single words/phrases.
    If unsure, guess the best you can.

    Text snippet:
    \"\"\"{text_snippet[:3000]}\"\"\"
    """

    try:
        openai.api_key = openai_api_key
        client = openai.Client(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0
        )
        content = resp.choices[0].message.content.strip()
        # Now parse comma-separated roles
        roles = [r.strip().lower() for r in content.split(",") if r.strip()]
        return roles
    except Exception as e:
        print("Error detecting roles with OpenAI:", e)
        return []

def ingest_supplier_pdf(
    db: Session,             # so we can store services in Postgres
    pipeline,               # RAGPipeline instance (for embeddings, collection)
    pdf_path: str,
    supplier_id: str,
    openai_api_key: str
):
    """
    1) Read & chunk PDF
    2) Embed in vector DB
    3) Use OpenAI to guess multiple roles -> store in Services & link in SupplierServices
    """
    # Step A: chunk PDF
    chunks = read_and_chunk_pdf_adaptive(pdf_path, max_words=150)
    print("DEBUG: Extracted chunks:", chunks)
    if not chunks:
        return "No text found"

    # Step B: remove old docs from the vector DB for that supplier
    pipeline.collection.delete_many({"supplier_id": supplier_id})

    # Step C: embed & store in vector DB
    embs = batch_embed_texts(pipeline.embedding_model, chunks, batch_size=16)
    docs_to_insert = []
    for i, chunk_text in enumerate(chunks):
        docs_to_insert.append({
            "supplier_id": supplier_id,
            "chunk_text": chunk_text,
            "embedding": embs[i].tolist()
        })
    if docs_to_insert:
        pipeline.collection.insert_many(docs_to_insert)

    # Step D: combine chunk text into a snippet for role detection
    combined_text = " ".join(chunks[:3])  # just first 3 chunks
    print("DEBUG: Combined text for role detection:", combined_text)
    roles_detected = detect_roles_from_text(combined_text, openai_api_key, num_roles=5)

    # Step E: store each role in Postgres & link to supplier
    for role in roles_detected:
        svc = repository.get_service_by_name(db, role)
        if not svc:
            # create new service
            svc_create = schemas.ServiceCreate(name=role, description="")
            svc = repository.create_service(db, svc_create)
        # link
        repository.link_supplier_service(db, supplier_id, svc.id)

    return f"Ingested {len(docs_to_insert)} chunks, detected roles: {roles_detected}"
