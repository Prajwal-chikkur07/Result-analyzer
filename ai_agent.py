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
    built from the currently loaded PDF. Enhanced to handle specific queries
    about failed/passed students with table formatting.
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    # Use knowledge base if available, otherwise fall back to passed context_data
    if _knowledge_base["raw_data"]:
        context = build_context()
        
        # Handle specific queries with direct data processing
        prompt_lower = prompt.lower()
        
        # Check for failed students query
        if any(word in prompt_lower for word in ['failed', 'fail', 'failing']):
            failed_students = [s for s in _knowledge_base["raw_data"] if s.get("Result", "").upper() == "FAIL"]
            if failed_students:
                response = f"Here are the students who failed:\n\n"
                response += "| Student Name | USN | Total Marks | Result |\n"
                response += "|--------------|-----|-------------|--------|\n"
                for student in failed_students:
                    name = student.get("Student Name", "N/A")
                    usn = student.get("USN", "N/A")
                    total = student.get("Total", "N/A")
                    result = student.get("Result", "N/A")
                    response += f"| {name} | {usn} | {total} | {result} |\n"
                response += f"\n**Total Failed Students: {len(failed_students)}**"
                
                # Add subject-wise failure analysis if available
                if 'subject' in prompt_lower:
                    subject_cols = _knowledge_base.get("subject_columns", [])
                    if subject_cols:
                        response += "\n\n**Subject-wise Failure Details:**\n"
                        for student in failed_students:
                            response += f"\n**{student.get('Student Name', 'N/A')}:**\n"
                            for subject in subject_cols:
                                if subject in student:
                                    score = student[subject]
                                    if isinstance(score, (int, float)) and score < 35:  # Assuming 35 is pass mark
                                        response += f"- {subject}: {score} (Failed)\n"
                
                return response
            else:
                return "Great news! No students have failed in this examination. All students have passed successfully! 🎉"
        
        # Check for passed students query
        elif any(word in prompt_lower for word in ['passed', 'pass', 'passing', 'successful']):
            passed_students = [s for s in _knowledge_base["raw_data"] if s.get("Result", "").upper() == "PASS"]
            if passed_students:
                response = f"Here are the students who passed:\n\n"
                response += "| Student Name | USN | Total Marks | Result |\n"
                response += "|--------------|-----|-------------|--------|\n"
                for student in passed_students:
                    name = student.get("Student Name", "N/A")
                    usn = student.get("USN", "N/A")
                    total = student.get("Total", "N/A")
                    result = student.get("Result", "N/A")
                    response += f"| {name} | {usn} | {total} | {result} |\n"
                response += f"\n**Total Passed Students: {len(passed_students)}**"
                
                # Add top performers if requested
                if any(word in prompt_lower for word in ['top', 'best', 'highest']):
                    top_students = sorted(passed_students, key=lambda x: x.get("Total", 0), reverse=True)[:5]
                    response += "\n\n**Top 5 Performers:**\n"
                    for i, student in enumerate(top_students, 1):
                        response += f"{i}. {student.get('Student Name', 'N/A')} - {student.get('Total', 'N/A')} marks\n"
                
                return response
            else:
                return "Unfortunately, no students have passed in this examination. This requires immediate attention and review of the assessment or teaching methods."
        
        # Check for subject-specific queries
        elif any(subject in prompt_lower for subject in ['math', 'physics', 'chemistry', 'english', 'biology']):
            subject_name = None
            for subject in ['math', 'physics', 'chemistry', 'english', 'biology']:
                if subject in prompt_lower:
                    subject_name = subject.title()
                    break
            
            if subject_name:
                # Find the actual subject column name
                subject_cols = _knowledge_base.get("subject_columns", [])
                actual_subject = None
                for col in subject_cols:
                    if subject_name.lower() in col.lower():
                        actual_subject = col
                        break
                
                if actual_subject:
                    response = f"**{actual_subject} Analysis:**\n\n"
                    
                    # Get subject-specific data
                    subject_data = []
                    for student in _knowledge_base["raw_data"]:
                        if actual_subject in student:
                            subject_data.append({
                                "name": student.get("Student Name", "N/A"),
                                "usn": student.get("USN", "N/A"),
                                "score": student.get(actual_subject, "N/A"),
                                "result": "Pass" if isinstance(student.get(actual_subject), (int, float)) and student.get(actual_subject, 0) >= 35 else "Fail"
                            })
                    
                    if subject_data:
                        response += "| Student Name | USN | Score | Status |\n"
                        response += "|--------------|-----|-------|--------|\n"
                        for data in subject_data:
                            response += f"| {data['name']} | {data['usn']} | {data['score']} | {data['result']} |\n"
                        
                        # Add statistics
                        scores = [d['score'] for d in subject_data if isinstance(d['score'], (int, float))]
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            max_score = max(scores)
                            min_score = min(scores)
                            passed_count = len([s for s in scores if s >= 35])
                            
                            response += f"\n**{actual_subject} Statistics:**\n"
                            response += f"- Average Score: {avg_score:.1f}\n"
                            response += f"- Highest Score: {max_score}\n"
                            response += f"- Lowest Score: {min_score}\n"
                            response += f"- Students Passed: {passed_count}/{len(scores)}\n"
                            response += f"- Pass Percentage: {(passed_count/len(scores)*100):.1f}%\n"
                    
                    return response
        
        # Check for topper/top students query
        elif any(word in prompt_lower for word in ['topper', 'top', 'best', 'highest', 'rank']):
            top_students = _knowledge_base.get("top_students", [])
            if top_students:
                response = "**Top Performing Students:**\n\n"
                response += "| Rank | Student Name | USN | Total Marks | Result |\n"
                response += "|------|--------------|-----|-------------|--------|\n"
                for i, student in enumerate(top_students, 1):
                    name = student.get("Student Name", "N/A")
                    usn = student.get("USN", "N/A")
                    total = student.get("Total", "N/A")
                    result = student.get("Result", "N/A")
                    response += f"| {i} | {name} | {usn} | {total} | {result} |\n"
                
                # Add class topper details
                if top_students:
                    topper = top_students[0]
                    response += f"\n🏆 **Class Topper:** {topper.get('Student Name', 'N/A')} with {topper.get('Total', 'N/A')} marks!"
                
                return response
        
        # Check for statistics/summary queries
        elif any(word in prompt_lower for word in ['statistics', 'summary', 'overview', 'total', 'count']):
            abstract = _knowledge_base.get("abstract", {})
            response = "**Examination Statistics Summary:**\n\n"
            response += f"📊 **Overall Performance:**\n"
            response += f"- Total Students: {abstract.get('total_students', 'N/A')}\n"
            response += f"- Students Passed: {abstract.get('passed_students', 'N/A')}\n"
            response += f"- Students Failed: {abstract.get('failed_students', 'N/A')}\n"
            response += f"- Pass Percentage: {abstract.get('pass_percentage', 'N/A')}%\n\n"
            
            # Add subject-wise summary
            subject_analysis = _knowledge_base.get("subject_analysis", [])
            if subject_analysis:
                response += "**Subject-wise Performance:**\n\n"
                response += "| Subject | Appeared | Passed | Failed | Pass % | Highest | Lowest |\n"
                response += "|---------|----------|--------|--------|--------|---------|--------|\n"
                for subject in subject_analysis:
                    response += f"| {subject.get('subject', 'N/A')} | {subject.get('appeared', 'N/A')} | {subject.get('passed', 'N/A')} | {subject.get('failed', 'N/A')} | {subject.get('pass_percentage', 'N/A')}% | {subject.get('highest', 'N/A')} | {subject.get('lowest', 'N/A')} |\n"
            
            return response
        
    elif context_data:
        context = json.dumps(context_data[:30], indent=2)
    else:
        return "I'm ready to help you analyze student performance data! Please upload a PDF result file so I can provide detailed insights about students, subjects, grades, and performance trends. Once you upload the data, I can answer questions like:\n\n• Who are the top performers?\n• Which students need help?\n• Subject-wise analysis\n• Pass/fail statistics\n• Individual student details\n\nJust upload your data and ask me anything!"

    # If no specific query matched, use the AI model
    system_prompt = (
        "You are an Intelligent Academic Result Analysis Agent. "
        "You provide helpful, conversational responses about student performance data. "
        "Always be encouraging and provide actionable insights. "
        "Format your responses in a friendly, chat-like manner with clear structure.\n\n"
        f"Data Context:\n{context}\n\n"
        "Instructions:\n"
        "- Always provide a helpful response, never say you cannot answer\n"
        "- Use bullet points and clear formatting\n"
        "- Be encouraging and positive\n"
        "- Provide specific numbers and insights from the data\n"
        "- If asked about failed/passed students, create a clear table format\n"
        "- End with a helpful suggestion or follow-up question"
    )

    payload = {
        "inputs": f"<s>[INST] {system_prompt}\n\nQuestion: {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.2,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                ai_response = result[0].get("generated_text", "").strip()
                if ai_response:
                    return ai_response
                else:
                    return generate_fallback_response(prompt)
            return generate_fallback_response(prompt)
        else:
            print(f"HF Error: {response.status_code} — {response.text}")
            return generate_fallback_response(prompt)
    except Exception as e:
        print(f"AI query error: {str(e)}")
        return generate_fallback_response(prompt)

def generate_fallback_response(prompt):
    """Generate intelligent fallback responses when AI service is unavailable"""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your AI Result Analysis Assistant. I'm here to help you understand student performance, analyze grades, and provide insights about academic results. What would you like to know?"
    
    elif any(word in prompt_lower for word in ['help', 'what can you do']):
        return "I can help you with:\n\n• 📊 Student performance analysis\n• 📈 Subject-wise breakdowns\n• 🏆 Top performer identification\n• ⚠️ Students needing support\n• 📋 Pass/fail statistics\n• 📊 Grade distributions\n• 🎯 Performance trends\n\nJust ask me anything about your student data!"
    
    elif any(word in prompt_lower for word in ['failed', 'fail']):
        return "I understand you want to know about students who need support. While I'm having trouble accessing the detailed data right now, I typically provide:\n\n• Complete list of students who failed\n• Subject-wise failure analysis\n• Specific areas needing improvement\n• Recommendations for intervention\n\nPlease try your question again, or check if the system is properly connected."
    
    elif any(word in prompt_lower for word in ['passed', 'pass']):
        return "Great question about successful students! I usually provide:\n\n• List of all students who passed\n• Top performers and their scores\n• Subject-wise pass rates\n• Achievement highlights\n\nI'm having trouble accessing your data right now. Please try again or ensure the backend service is running."
    
    else:
        return f"I'd love to help you with '{prompt}'! I'm designed to provide comprehensive analysis of student performance data. While I'm experiencing some technical difficulties accessing your specific data right now, I'm equipped to handle all kinds of academic questions.\n\nPlease try your question again, or feel free to ask about:\n• Student performance\n• Subject analysis\n• Grade statistics\n• Academic insights"
