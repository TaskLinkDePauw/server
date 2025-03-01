import openai
from functools import lru_cache
from ..log_util import log_info, log_warning, log_error

@lru_cache(maxsize=128)
def generate_multi_queries(user_query, num_queries=3, openai_api_key=None, log_event_fn=None):
    """
    Use OpenAI or any LLM to generate multiple variants of the user query.
    """
    if not openai_api_key:
        return [user_query]

    try:
        client = openai.Client(api_key=openai_api_key)
        prompt = f"""
        You are an AI assistant. Given the user's query:
        "{user_query}"

        Generate {num_queries} alternative search queries or rephrasings
        that might retrieve relevant but slightly different results. 
        Separate each query by a newline.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful query rewriter."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
        multi_queries_raw = response.choices[0].message.content.strip().split("\n")
        multi_queries = [q.strip() for q in multi_queries_raw if q.strip()]

        if log_event_fn:
            log_event_fn("MultiQueryGenerated", {"original_query": user_query, "queries": multi_queries})
        else:
            # Default to log_info if no log_event_fn is provided
            log_info("MultiQueryGenerated", f"Original: {user_query}, Queries: {multi_queries}")

        return multi_queries
    except openai.error.OpenAIError as e:
        log_error("OpenAIError", f"generate_multi_queries: {str(e)}")
        return [user_query]
    except Exception as e:
        log_error("UnexpectedError", f"generate_multi_queries: {str(e)}")
        return [user_query]
