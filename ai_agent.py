import requests
import json
import os

HF_API_KEY = os.environ.get("HF_API_KEY", "your-huggingface-token-here")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Global knowledge base — rebuilt every time a new PDF is processed
_knowledge_base = {
    "raw_data": [],
    "abstract": {},
    "subject_analysis": [],
    "top_students": [],
    "pdf_filename": ""
}

def update_knowledge_base(analysis_result, pdf_filename=""):
    """
    Called after every PDF upload to rebuild the agent's context.
    """
    _knowledge_base["raw_data"]         = analysis_result.get("raw_data", [])
    _knowledge_base["abstract"]         = analysis_result.get("abstract", {})
    _knowledge_base["subject_analysis"] = analysis_result.get("subject_analysis", [])
    _knowledge_base["top_students"]     = analysis_result.get("top_students", [])
    _knowledge_base["subject_columns"]  = analysis_result.get("subject_columns", [])
    _knowledge_base["pdf_filename"]     = pdf_filename
    print(f"[AI Agent] Knowledge base updated for '{pdf_filename}' — "
          f"{len(_knowledge_base['raw_data'])} students, subjects: {_knowledge_base['subject_columns']}")

def build_context():
    """
    Builds a comprehensive structured context from the currently loaded PDF.
    Includes per-student subject marks so the LLM can answer subject-specific questions.
    """
    ab   = _knowledge_base["abstract"]
    subs = _knowledge_base["subject_analysis"]
    top  = _knowledge_base["top_students"]
    raw  = _knowledge_base["raw_data"]
    sub_cols = _knowledge_base["subject_columns"]

    context  = f"=== PDF SOURCE: {_knowledge_base['pdf_filename']} ===\n\n"
    context += "--- RESULT SUMMARY ---\n"
    context += f"Total Students : {ab.get('total_students', 0)}\n"
    context += f"Passed         : {ab.get('passed_students', 0)}\n"
    context += f"Failed         : {ab.get('failed_students', 0)}\n"
    context += f"Pass Percentage: {ab.get('pass_percentage', 0)}%\n\n"

    context += "--- SUBJECT-WISE STATISTICS ---\n"
    for s in subs:
        context += (f"  {s['subject']}: "
                    f"Appeared={s['appeared']}, Passed={s['passed']}, Failed={s['failed']}, "
                    f"Pass%={s['pass_percentage']}%, Highest={s['highest']}, "
                    f"Lowest={s['lowest']}, Avg={s.get('average','N/A')}\n")

    context += "\n--- TOP 5 STUDENTS ---\n"
    for i, s in enumerate(top, 1):
        context += (f"  {i}. {s.get('Student Name','?')} | USN: {s.get('USN','?')} | "
                    f"Total: {s.get('Total',0)} | {s.get('Result','?')}\n")

    context += f"\n--- ALL STUDENT RECORDS (Subject columns: {', '.join(sub_cols)}) ---\n"
    for r in raw[:50]:
        marks_str = " | ".join(
            f"{col}: {r.get(col, '-')}" for col in sub_cols if col in r
        )
        context += (f"  {r.get('Student Name','?')} | USN: {r.get('USN','?')} | "
                    f"{marks_str} | Total: {r.get('Total',0)} | Result: {r.get('Result','?')}\n")

    return context

def query_hf(prompt, context_data=None):
    """
    Queries Hugging Face Inference API with rich structured context
    built from the currently loaded PDF.
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    # Use knowledge base if available, otherwise fall back to passed context_data
    if _knowledge_base["raw_data"]:
        context = build_context()
    elif context_data:
        context = json.dumps(context_data[:30], indent=2)
    else:
        return "No student data loaded. Please upload a PDF first."

    system_prompt = (
        "You are an Intelligent Academic Result Analysis Agent. "
        "You have been trained on the following examination result data extracted from the uploaded PDF. "
        "Answer ONLY based on this data. Be specific, accurate, and professional.\n\n"
        f"{context}"
    )

    payload = {
        "inputs": f"<s>[INST] {system_prompt}\n\nQuestion: {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.1,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "I couldn't generate a response.").strip()
            return "Unexpected response format from AI model."
        else:
            print(f"HF Error: {response.status_code} — {response.text}")
            return f"AI service error (HTTP {response.status_code}). Please retry."
    except Exception as e:
        return f"AI query failed: {str(e)}"
