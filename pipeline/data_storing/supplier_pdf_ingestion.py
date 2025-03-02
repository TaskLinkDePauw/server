import os
import re
import openai

from .chunking_utils import read_and_chunk_pdf_adaptive
from .embedding_utils import batch_embed_texts
from ..log_util import log_event

def detect_role_from_text(text_snippet: str, openai_api_key: str) -> str:
    """
    Calls OpenAI (GPT) to guess the main role from the PDF text.
    Returns something like 'personal trainer', 'plumber', or 'unknown' if uncertain.
    """
    if not openai_api_key:
        return "unknown"

    prompt = f"""
    The following text is from a resume or profile. Based on the text, infer the single best job role or profession.
    Text snippet:
    \"\"\"{text_snippet[:2000]}\"\"\"

    Return a short, lowercase label like "personal trainer", "plumber", "graphic designer", "nail technician", etc.
    If unsure, return "unknown".
    """

    try:
        # Ensure the global openai.api_key is set:
        openai.api_key = openai_api_key

        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        role_guess = resp.choices[0].message.content.strip().lower()
        # Basic sanitize: remove punctuation above letters/spaces
        role_guess = re.sub(r"[^\w\s]", "", role_guess).strip()
        # If it’s too long, treat as unknown:
        if len(role_guess) > 30:
            role_guess = "unknown"
        return role_guess

    except Exception as e:
        log_event("RoleDetectionError", str(e))
        return "unknown"


def ingest_supplier_pdf(
    pipeline,       # an instance of your RAGPipeline
    pdf_path: str,
    supplier_id: str
) -> str:
    """
    1) Chunk the PDF
    2) Auto-detect service_role from first chunk(s) using LLM
    3) Delete old docs for this supplier (re-upload scenario)
    4) Embed & store new chunk docs in Mongo with supplier_id & service_role
    Returns the detected role (string).
    """

    # 1) Chunk
    chunks = read_and_chunk_pdf_adaptive(pdf_path, max_words=150)
    if not chunks:
        log_event("EmptyPDF", f"No text found in PDF {pdf_path}")
        return "unknown"

    # 2) Auto-detect role
    if pipeline.openai_api_key:
        combined_text = " ".join(chunks[:3])  # short snippet
        service_role = detect_role_from_text(combined_text, pipeline.openai_api_key)
    else:
        service_role = "unknown"

    # 3) Remove old docs for this supplier to handle re-embedding
    pipeline.collection.delete_many({"supplier_id": supplier_id})

    # 4) Embed in batches
    embeddings = batch_embed_texts(pipeline.embedding_model, chunks, batch_size=16)

    # Construct docs
    docs_to_insert = []
    for idx, chunk_text in enumerate(chunks):
        doc = {
            "_id": f"{supplier_id}_chunk_{idx}",
            "supplier_id": supplier_id,
            "service_role": service_role,
            "chunk_text": chunk_text,
            "embedding": embeddings[idx].tolist(),
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        pipeline.collection.insert_many(docs_to_insert)
        log_event(
            "IngestedSupplierPDF",
            f"Ingested {len(docs_to_insert)} docs for supplier_id={supplier_id}, role={service_role}"
        )

    return service_role
