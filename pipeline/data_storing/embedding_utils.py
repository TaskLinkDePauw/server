import os
import uuid
from .chunking_utils import read_and_chunk_pdf_adaptive
from ..log_util import log_info, log_warning, log_error

def store_role_if_missing(db, role_name):
    """
    Check if 'role_name' exists in the 'roles' collection.
    If not, insert a new doc with a default or generated description.
    """
    try:
        roles_coll = db["roles"]
        existing = roles_coll.find_one({"role_name": role_name})
        if not existing:
            # Basic or LLM-based approach
            new_description = f"You are a {role_name}. You handle tasks related to {role_name}."
            doc = {
                "role_name": role_name,
                "role_description": new_description
            }
            roles_coll.insert_one(doc)
            log_info("RoleInserted", f"Inserted new role: {role_name}")
    except Exception as e:
        log_error("RoleInsertError", f"store_role_if_missing: {str(e)}")

def batch_embed_texts(embedding_model, texts, batch_size=16):
    """
    Embeds a list of texts in batches for more efficient encoding.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_embs = embedding_model.encode(batch_texts)
        except Exception as e:
            log_error("EmbeddingError", f"batch_embed_texts failed: {str(e)}")
            # fallback: skip or re-try logic
            batch_embs = []
        all_embeddings.extend(batch_embs)
    return all_embeddings

def embed_and_store_in_batches(pipeline, pdf_files, batch_size=16, max_words=150):
    """
    Variant using batch embedding for efficiency.
    """
    try:
        pipeline.collection.delete_many({})
    except Exception as e:
        log_error("MongoDBDeleteError", f"delete_many failed: {str(e)}")
        return

    all_docs = []
    for pdf_path in pdf_files:
        # Extract role from the pdf file's path
        sanitized_role = pdf_path.split(os.sep)[-2]
        role = sanitized_role.replace("_", " ")
        
        # Ensure the role is stored in the DB if it's missing
        store_role_if_missing(pipeline.db, role)
        
        # Chunk & embed
        chunks = read_and_chunk_pdf_adaptive(pdf_path, max_words=max_words)
        chunk_texts = list(chunks)

        embeddings = batch_embed_texts(pipeline.embedding_model, chunk_texts, batch_size=batch_size)

        for idx, emb in enumerate(embeddings):
            doc = {
                "_id": f"{os.path.basename(pdf_path)}_chunk_{uuid.uuid4()}",
                "pdf_file": os.path.basename(pdf_path),
                "chunk_text": chunk_texts[idx],
                "embedding": emb.tolist(),
                "role": role,
                "embedding_model_name": pipeline.embedding_model.__class__.__name__
            }
            all_docs.append(doc)

    if all_docs:
        try:
            pipeline.collection.insert_many(all_docs)
            log_info("EmbeddingsInserted", f"Inserted {len(all_docs)} chunk-documents.")
        except Exception as e:
            log_error("MongoDBInsertError", f"insert_many failed: {str(e)}")
