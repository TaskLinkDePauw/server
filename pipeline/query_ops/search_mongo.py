# pipeline/search_mongo.py

def search_mongo(
    collection,
    embedding_model,
    query_text,
    top_k=3,
    roles_filter=None,
    index_name="default"
):
    """
    Vector search in MongoDB Atlas for a single query.
    """
    query_emb = embedding_model.encode(query_text).tolist()

    vector_search_stage = {
        "$vectorSearch": {
            "index": index_name,
            "queryVector": query_emb,
            "path": "embedding",
            "limit": top_k * 2,
            "numCandidates": 50
        }
    }

    pipeline = [vector_search_stage]

    if roles_filter and roles_filter != "all":
        if isinstance(roles_filter, set):
            match_condition = {"role": {"$in": list(roles_filter)}}
        elif isinstance(roles_filter, str):
            match_condition = {"role": roles_filter}
        pipeline.append({"$match": match_condition})

    pipeline.append({
        "$project": {
            "_id": 1,
            "supplier_id": 1,    # If you store it
            "pdf_file": 1,
            "chunk_text": 1,
            "score": {"$meta": "vectorSearchScore"},
            "service_role": 1
        }
    })

    results = list(collection.aggregate(pipeline))
    return results
