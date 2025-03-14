##########################################################
# pipeline/structured_output/structured_out.py
##########################################################
import openai
from typing import Optional
from .log_util import log_info, log_error

def build_summary_prompt(user_query, docs):
    combined_docs = "\n\n".join([f"- {d['chunk_text']}" for d in docs])
    few_shot_example = """
    Example Q: "I need someone with plumbing experience."
    Example A:
    Candidate Name: John Smith
    Key Strengths: Pipe installation, fixture repairs
    Reasoning: They have proven plumbing experience from past roles
    """
    prompt = f"""
    You are a helpful AI that reads candidate resumes. Use the context below
    to answer the user's query in a structured way.

    [Few Shot Example]
    {few_shot_example}

    [User Query]
    {user_query}

    [Context]
    {combined_docs}

    Please provide an answer with:
    1) Candidate Name (if known)
    2) Key Strengths
    3) Reasoning for why they match the query
    """
    return prompt

def ask_chatgpt_structured(user_query, retrieved_docs, openai_api_key=None, method="pydantic", log_event_fn=None):
    if not retrieved_docs:
        return {
            "candidate_name": "",
            "key_strengths": [],
            "reasoning": "No documents found."
        }

    if method == "function_calling":
        functions = [
            {
                "name": "recommend_candidate",
                "description": "Return structured info about candidate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_name": {"type": "string"},
                        "key_strengths": {"type": "array", "items": {"type": "string"}},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["candidate_name", "key_strengths", "reasoning"]
                },
            }
        ]

        prompt_content = build_summary_prompt(user_query, retrieved_docs)

        try:
            client = openai.Client(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
                functions=functions,
                function_call={"name": "recommend_candidate"}
            )
            message = response.choices[0].message
            fc = getattr(message, "function_call", None)
            if fc is None:
                structured_json = None
            elif isinstance(fc, dict):
                structured_json = fc.get("arguments")
            else:
                structured_json = getattr(fc, "arguments", None)

            if log_event_fn:
                log_event_fn("StructuredOutputFunctionCall", structured_json)
            else:
                log_info("StructuredOutputFunctionCall", str(structured_json))

            return structured_json
        except Exception as e:
            if log_event_fn:
                log_event_fn("StructuredOutputError", str(e))
            else:
                log_error("StructuredOutputError", str(e))
            return None

    else:
        # pydantic approach
        prompt_content = build_summary_prompt(user_query, retrieved_docs)
        pydantic_instructions = """
        Please return valid JSON with the following keys:
        {
            "candidate_name": "<string>",
            "key_strengths": ["<string>", "<string>"],
            "reasoning": "<string>"
        }
        """
        full_prompt = prompt_content + "\n" + pydantic_instructions

        try:
            client = openai.Client(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7
            )
            content = response.choices[0].message.content
            import json
            structured_data = json.loads(content)

            if log_event_fn:
                log_event_fn("StructuredOutputPydantic", structured_data)
            else:
                log_info("StructuredOutputPydantic", str(structured_data))

            return structured_data
        except Exception as e:
            if log_event_fn:
                log_event_fn("StructuredOutputError", str(e))
            else:
                log_error("StructuredOutputError", str(e))
            return None

def summarize_uploaded_pdf_for_supplier(
    pipeline,
    supplier_id: str,
    openai_api_key: Optional[str] = None,
    max_tokens: int = 512
) -> str:
    """
    Summarizes the content of the uploaded PDF(s) for a given supplier:
    - Gathers all 'chunk_text' from pipeline.collection (Mongo) 
      where supplier_id = supplier_id.
    - Concatenates them into one text snippet (with a size cap).
    - Calls OpenAI to produce a bullet-point style summary focusing on 
      the supplier's jobs, skills, and professional experience.

    Args:
        pipeline: An instance of your RAGPipeline or a similar object 
                  that has 'collection' and possibly openai_api_key if not provided.
        supplier_id: The ID of the supplier in your system (string/UUID).
        openai_api_key: An optional API key if not using pipeline's stored key.
        max_tokens: The maximum tokens or length for the final summary 
                    (not strictly enforced, but used to shape the prompt).

    Returns:
        A string containing the summarized content. If something goes wrong,
        returns an error message or empty string.
    """
    # 1) Check if we have an API key
    final_api_key = openai_api_key or getattr(pipeline, "openai_api_key", None)
    if not final_api_key:
        log_error("SummaryError", "No OpenAI API key provided.")
        return "No OpenAI API key available; cannot summarize."

    # 2) Collect all chunk texts for this supplier
    docs_cursor = pipeline.collection.find({"supplier_id": supplier_id}, {"_id": 0, "chunk_text": 1})
    all_chunks = [doc["chunk_text"] for doc in docs_cursor]
    if not all_chunks:
        return "No PDF content found for this supplier."

    # Combine them into one large text snippet
    combined_text = "\n".join(all_chunks)

    # Optionally, to avoid hitting token limits, you can truncate
    # the text to a certain number of words:
    words = combined_text.split()
    # e.g., limit to ~2000 words
    max_words = 2000
    if len(words) > max_words:
        combined_text = " ".join(words[:max_words]) + "\n...\n[Truncated]"
    
    # 3) Build a prompt for summarizing
    prompt = f"""
    You are an AI that summarizes resumes or profiles.

    Please provide a succinct summary focusing on:
    - The supplier's main job roles
    - Their key skills
    - Their professional experience

    The text is (some may be truncated):
    \"\"\"{combined_text}\"\"\"

    Output a bullet-point style summary (in plain text), 
    highlighting the supplier's job roles, top skills, 
    and relevant experience.
    """

    # 4) Call OpenAI to get the summary
    try:
        openai.api_key = final_api_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful resume summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )
        summary_text = resp.choices[0].message.content.strip()
        log_info("SummarizeUploadedPDF", f"Supplier {supplier_id} summary generated.")
        return summary_text
    except Exception as e:
        log_error("SummarizeError", str(e))
        return f"Error summarizing PDF content: {e}"
