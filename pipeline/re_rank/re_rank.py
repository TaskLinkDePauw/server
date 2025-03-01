# pipeline/re_rank.py
import openai
from ..log_util import log_info, log_error, log_event
def re_rank_results(user_query, results, method="reciprocal_rank_fusion", top_k=3, log_event_fn=None, openai_api_key=None):
    """
    Re-rank or fuse the results. We can choose among:
        - 'cohere' (Cohere ReRank)
        - 'openai' (OpenAI prompt-based re-rank)
        - 'reciprocal_rank_fusion' (heuristic)
    Returns final top_k results in a list.
    """
    if not results:
        return []

    if method == "cohere":
        # Example (requires `pip install cohere` and a Cohere API key)
        # co = cohere.Client(os.getenv("COHERE_API_KEY"))
        # docs = [res["chunk_text"] for res in results]
        # re_ranked = co.rerank(query=user_query, documents=docs, top_n=len(results))
        # # "re_ranked" is a list with .index, .relevance_score
        # # You can then reorder 'results' accordingly
        # # For demonstration, we do a stub:
        log_event("ReRank", "Cohere ReRank not implemented in detail here.")
        # Return as is for now:
        final = results[:top_k]
        return final

    elif method == "openai":
        # Prompt-based re-rank approach
        # We'll do a simple example of asking GPT to reorder them.
        # More advanced approach might store chunk_text in a single prompt
        # and parse the output. This is a stub for demonstration.
        content_for_rerank = ""
        for i, r in enumerate(results):
            content_for_rerank += f"Document {i+1}: {r['chunk_text']}\n"
        re_rank_prompt = f"""
        The user's query is: "{user_query}".
        The following are some candidate text chunks. 
        Please rank them from most relevant to least relevant for the query:

        {content_for_rerank}

        Output the sorted document numbers in order of relevance, most relevant first.
        """
        try:
            client = openai.Client(api_key=openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": re_rank_prompt}],
                temperature=0
            )
            ranking_text = resp.choices[0].message.content
            # You'd parse that ranking_text to reorder `results`.
            # For demonstration, we'll just do a direct slice:
            final = results[:top_k]
            return final
        except openai.error.OpenAIError as e:
            log_error("OpenAIError", f"re_rank_results: {str(e)}")
            return results[:top_k]
        except Exception as e:
            log_error("UnexpectedError", f"re_rank_results: {str(e)}")
            return results[:top_k]

    else:
        # Reciprocal Rank Fusion or simple heuristic
        # For a single result list, we'll just do top_k by score desc
        # If we had multiple lists, we'd do the fusion. Here is a simple approach:
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]
