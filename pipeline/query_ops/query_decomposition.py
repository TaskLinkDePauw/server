##########################################################
# pipeline/query_ops/query_decomposition.py
##########################################################
import openai
from ..log_util import log_info, log_error

def decompose_query(user_query, openai_api_key=None, log_event_fn=None):
    """
    Break a complex user query into sub-queries.
    """
    if not openai_api_key:
        return [user_query]

    decomposition_prompt = f"""
    You are a helpful assistant. 
    The user query is: '{user_query}'
    Break this query into 2-4 smaller sub-queries or aspects, each focusing on a distinct requirement.
    Return them each on a separate line.
    """
    try:
        client = openai.Client(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": decomposition_prompt}],
            temperature=0.7
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        sub_queries = [l.strip() for l in lines if l.strip()]

        if log_event_fn:
            log_event_fn("QueryDecomposed", {"original_query": user_query, "sub_queries": sub_queries})
        else:
            log_info("QueryDecomposed", f"Original: {user_query}, Sub-queries: {sub_queries}")

        return sub_queries
    except openai.error.OpenAIError as e:
        log_error("OpenAIError", f"decompose_query: {str(e)}")
        return [user_query]
    except Exception as e:
        log_error("UnexpectedError", f"decompose_query: {str(e)}")
        return [user_query]
