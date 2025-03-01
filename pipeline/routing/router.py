import openai
from ..log_util import log_info, log_warning, log_error



def route_query_llm(user_query, openai_api_key=None, role_templates=None):
    if not openai_api_key:
        # fallback
        return

    system_prompt = (
        "You are an expert in routing user queries to the appropriate role. "
        "We have these roles: software developer, plumber, electrician, barber, personal trainer. "
        "If the query does not fit one of these, respond with 'all'. "
        "Return just the role name or 'all'."
    )
    user_message = f"User query: {user_query}"
    try:
        print("Routing query using LLM...")
        client = openai.Client(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        choice = resp.choices[0].message.content.strip().lower()
        valid_roles = set(role_templates.keys()) if role_templates else set()

        if choice in valid_roles:
            return choice
        else:
            return "all"
    except Exception as e:
        log_error("LLMRoutingError", str(e))
        return "all"
