import requests
import json
import os

# Google Gemini API configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyC3Ys3cy_ewpAfYee8_-XbKeJv38Q44hv8")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_API_KEY}"

# Fallback to Hugging Face if needed
HF_API_KEY = os.environ.get("HF_API_KEY", "your-huggingface-token-here")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Global knowledge base — rebuilt every time a new PDF is processed
_knowledge_base = {
    "raw_data": [],
    "abstract": {},
    "subject_analysis": [],
    "top_students": [],
    "pdf_filename": ""
}

def query_gemini_api(prompt, context):
    """Query Google Gemini API for AI responses"""
    try:
        system_prompt = f"""You are an AI assistant for student result analysis. Provide direct, simple answers without extra formatting or explanations.

Data Context:
{context}

Instructions:
- Give minimal, direct responses
- No bullet points, emojis, or extra formatting
- For topper queries: just return "Name - Marks"
- For failed/passed students: return names or simple table
- For statistics: return basic numbers only
- Keep all responses short and to the point"""

        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\nQuestion: {prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 800,
            }
        }

        response = requests.post(GEMINI_API_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0].get("content", {})
                if "parts" in content and len(content["parts"]) > 0:
                    return content["parts"][0].get("text", "").strip()
        else:
            print(f"Gemini API Error: {response.status_code} — {response.text}")
            
    except Exception as e:
        print(f"Gemini API query error: {str(e)}")
    
    return None

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

    return context

def query_hf(prompt, context_data=None):
    """
    Enhanced AI query function that provides intelligent responses using Google Gemini API
    with fallback to direct data analysis when external services are unavailable.
    """
    # Use knowledge base if available, otherwise fall back to passed context_data
    if _knowledge_base["raw_data"]:
        context = build_context()
        
        # Handle specific queries with direct data processing
        prompt_lower = prompt.lower()
        
        # Check for topper/top students query - most common query
        if any(word in prompt_lower for word in ['topper', 'top', 'best', 'highest', 'rank']):
            top_students = _knowledge_base.get("top_students", [])
            if top_students:
                topper = top_students[0]
                name = topper.get("Student Name", "N/A")
                total = topper.get("Total", "N/A")
                return f"{name} - {total}"
            else:
                return "No data"
        
        # Check for failed students query
        elif any(word in prompt_lower for word in ['failed', 'fail', 'failing']):
            failed_students = [s for s in _knowledge_base["raw_data"] if s.get("Result", "").upper() == "FAIL"]
            if failed_students:
                if len(failed_students) <= 2:
                    names = [s.get("Student Name", "N/A") for s in failed_students]
                    return ", ".join(names)
                else:
                    response = "| Name | USN | Marks |\n|------|-----|-------|\n"
                    for student in failed_students:
                        name = student.get("Student Name", "N/A")
                        usn = student.get("USN", "N/A")
                        total = student.get("Total", "N/A")
                        response += f"| {name} | {usn} | {total} |\n"
                    return response
            else:
                return "None"
        
        # Check for passed students query
        elif any(word in prompt_lower for word in ['passed', 'pass', 'passing', 'successful']):
            passed_students = [s for s in _knowledge_base["raw_data"] if s.get("Result", "").upper() == "PASS"]
            if passed_students:
                if len(passed_students) <= 2:
                    names = [s.get("Student Name", "N/A") for s in passed_students]
                    return ", ".join(names)
                else:
                    response = "| Name | USN | Marks |\n|------|-----|-------|\n"
                    for student in passed_students:
                        name = student.get("Student Name", "N/A")
                        usn = student.get("USN", "N/A")
                        total = student.get("Total", "N/A")
                        response += f"| {name} | {usn} | {total} |\n"
                    return response
            else:
                return "None"
        
        # Check for statistics/summary queries
        elif any(word in prompt_lower for word in ['statistics', 'summary', 'overview', 'total', 'count']):
            abstract = _knowledge_base.get("abstract", {})
            total = abstract.get('total_students', 'N/A')
            passed = abstract.get('passed_students', 'N/A')
            failed = abstract.get('failed_students', 'N/A')
            percentage = abstract.get('pass_percentage', 'N/A')
            return f"Total: {total}, Passed: {passed}, Failed: {failed}, Pass Rate: {percentage}%"
        
        # For any other query, try Google Gemini API
        else:
            ai_response = query_gemini_api(prompt, context)
            if ai_response:
                return ai_response
            else:
                # Generate simple fallback response
                return f"I can help with questions about student performance. Try asking about toppers, failed students, or statistics."
        
    elif context_data:
        return generate_intelligent_response(prompt, {"raw_data": context_data})
    else:
        return "Upload a PDF file to get started."

def generate_intelligent_response(prompt, knowledge_base):
    """Generate intelligent responses based on available data"""
    prompt_lower = prompt.lower()
    
    # Get basic stats
    raw_data = knowledge_base.get("raw_data", [])
    abstract = knowledge_base.get("abstract", {})
    
    if not raw_data:
        return f"I'd love to help you with '{prompt}', but I don't have any student data loaded yet. Please upload a PDF result file first!"
    
    # General academic questions
    if any(word in prompt_lower for word in ['how many', 'total', 'count']):
        total_students = len(raw_data)
        passed = len([s for s in raw_data if s.get("Result", "").upper() == "PASS"])
        failed = total_students - passed
        
        return f"Based on the current data:\n\n• Total Students: {total_students}\n• Students Passed: {passed}\n• Students Failed: {failed}\n• Pass Percentage: {(passed/total_students*100):.1f}%\n\nWould you like more detailed information about any specific aspect?"
    
    elif any(word in prompt_lower for word in ['average', 'mean']):
        scores = [s.get("Total", 0) for s in raw_data if isinstance(s.get("Total"), (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
            return f"The average total score is {avg_score:.1f} marks.\n\nHighest score: {max(scores)}\nLowest score: {min(scores)}\n\nWould you like to see subject-wise averages?"
        
    elif any(word in prompt_lower for word in ['help', 'what can you do']):
        return f"I can help you analyze the {len(raw_data)} student records I have loaded! Here's what I can do:\n\n• 📊 Identify top performers and struggling students\n• 📈 Provide subject-wise analysis\n• 📋 Show pass/fail statistics\n• 🎯 Answer specific questions about grades\n• 📊 Generate performance insights\n\nJust ask me anything about the student data!"
    
    # Default response for unrecognized queries
    return f"I understand you're asking about '{prompt}'. While I have {len(raw_data)} student records loaded, I need a bit more specific information to give you the best answer.\n\nYou can ask me about:\n• Specific students or their performance\n• Subject-wise analysis\n• Pass/fail statistics\n• Top performers\n• Grade distributions\n\nCould you rephrase your question or be more specific about what you'd like to know?"