import os
import uuid
import openai
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer

# Local imports
from .log_utils import log_event
from .data_storing.chunking_utils import read_and_chunk_pdf_adaptive
from .data_storing.embedding_utils import embed_and_store, embed_and_store_in_batches, batch_embed_texts
from .routing.router import route_query_keyword, route_query_llm, route_query_semantic
from .re_rank.re_rank import re_rank_results
from .output.structured_output import build_summary_prompt, ask_chatgpt_structured
from .query_ops.multi_query import generate_multi_queries
from .query_ops.query_decomposition import decompose_query
from .query_ops.search_mongo import search_mongo

class RAGPipeline:
    """
    An upgraded pipeline that supports:
    - Multi-Query Generation
    - Re-ranking / Fusion
    - Routing by job role
    - Query Decomposition
    - Prompt Templates & Few-Shot
    - Structured Output
    - Observability (logging)

    pass: xl0DtM57EPvqFdoh
    """
    def __init__(
        self,
        mongo_uri,
        db_name,
        collection_name,
        index_name="default",
        openai_api_key=None,
        routing_method="keyword"
    ):
        self.mongo_uri = mongo_uri
        self.client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        try:
            self.client.admin.command('ping')
            log_event("MongoDBConnected", "Successfully connected to MongoDB!")
        except Exception as e:
            log_event("MongoDBConnectionError", str(e))

        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.index_name = index_name

        # Embedding model
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = 384

        # Decide which routing approach to use
        self.routing_method = routing_method

        # load roles from DB
        self.role_templates = self.load_roles_from_db()
        
        if openai_api_key:
            # Fix potential encoding issues in API key
            self.openai_api_key = openai_api_key.encode("utf-8").decode("utf-8")
            openai.api_key = self.openai_api_key
        else:
            self.openai_api_key = None
    
    ########################################################
    # PDF Ingestion & Chunking
    ########################################################
    def embed_and_store_in_batches(self, pdf_files, batch_size=16):
        """
        Variant using batch embedding for efficiency.
        (Delegates to the function in storing/embedding_utils.py)
        """
        return embed_and_store_in_batches(self, pdf_files, batch_size=batch_size)


    ########################################################
    # Routing
    ########################################################
    def route_query(self, user_query):
        """
        Decide which role(s) best matches the user query.
        """
        return route_query_llm(user_query, openai_api_key=self.openai_api_key, role_templates=self.role_templates)

    ########################################################
    # Query Decomposition
    ########################################################
    def decompose_query(self, user_query):
        return decompose_query(
            user_query=user_query,
            openai_api_key=self.openai_api_key,
            log_event_fn=log_event
        )
    
    ########################################################
    # Multi-Query Generation
    ########################################################
    def generate_multi_queries(self, user_query, num_queries=3):
        return generate_multi_queries(
            user_query=user_query,
            num_queries=num_queries,
            openai_api_key=self.openai_api_key,
            log_event_fn=log_event
        )
    
    ########################################################
    # Vector Search in MongoDB
    ########################################################
    def search_mongo(self, query_text, top_k=3, roles_filter=None):
        return search_mongo(
            collection=self.collection,
            embedding_model=self.embedding_model,
            query_text=query_text,
            top_k=top_k,
            roles_filter=roles_filter,
            index_name=self.index_name
        )
    
    ########################################################
    # Re-ranking / Fusion Helper
    ########################################################
    def re_rank_results(self, user_query,results, method="reciprocal_rank_fusion", top_k=3):
        return re_rank_results(
            user_query=user_query,
            results=results,
            method=method,
            top_k=top_k,
            log_event_fn=log_event, 
            openai_api_key=self.openai_api_key
        )
    
    ########################################################
    # Prompt Templates & Structured Output
    ########################################################
    def build_summary_prompt(self, user_query, docs):
        return build_summary_prompt(user_query, docs)

    def ask_chatgpt_structured(self, user_query, retrieved_docs, method="pydantic"):
        return ask_chatgpt_structured(
            user_query=user_query,
            retrieved_docs=retrieved_docs,
            openai_api_key=self.openai_api_key,
            method=method,
            log_event_fn=log_event
        )

    def rag_search(self, user_query, top_k=3, re_rank_method=None):
        log_event("SearchStart", user_query)

        # 1) Routing
        roles = self.route_query(user_query)
        log_event("Routing", f"Chosen roles: {roles}")

        # 2) Query Decomposition
        sub_queries = self.decompose_query(user_query)
        # (We already log_event inside decompose_query, but you could do more logging here if desired)

        # 3) Multi-Query Generation (just for the first sub-query)
        expanded_queries = self.generate_multi_queries(sub_queries[0], num_queries=2)

        # 4) Retrieve documents
        all_results = []
        for eq in expanded_queries:
            partial = self.search_mongo(eq, top_k=top_k, roles_filter=roles)
            all_results.extend(partial)

        # 5) Re-rank / fuse
        if re_rank_method:
            final_results = self.re_rank_results(user_query, all_results, method=re_rank_method, top_k=top_k)
        else:
            sorted_res = sorted(all_results, key=lambda x: x["score"], reverse=True)
            final_results = sorted_res[:top_k]

        log_event("SearchDone", f"Final result count: {len(final_results)}")
        return final_results
    
    def search_supplier(self, user_query, top_k=3, re_rank_method="reciprocal_rank_fusion"):
        log_event("SearchStart", user_query)

        # 1) Routing
        roles = self.route_query(user_query)
        log_event("Routing", f"Chosen roles: {roles}")

        # 2) Query Decomposition
        sub_queries = self.decompose_query(user_query)
        # (We already log_event inside decompose_query, but you could do more logging here if desired)

        # 3) Multi-Query Generation (just for the first sub-query)
        expanded_queries = self.generate_multi_queries(sub_queries[0], num_queries=2)

        # 4) Retrieve documents
        all_results = []
        for eq in expanded_queries:
            partial = self.search_mongo(eq, top_k=top_k, roles_filter=roles)
            all_results.extend(partial)

        # 5) Re-rank / fuse
        if re_rank_method:
            final_results = self.re_rank_results(user_query, all_results, method=re_rank_method, top_k=top_k)
        else:
            sorted_res = sorted(all_results, key=lambda x: x["score"], reverse=True)
            final_results = sorted_res[:top_k]

        log_event("SearchDone", f"Final result count: {len(final_results)}")
        return final_results
    