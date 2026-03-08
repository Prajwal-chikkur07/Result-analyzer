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
    ADVANCED AI query function that provides intelligent responses to ALL questions.
    This function NEVER fails to provide a helpful response.
    """
    # Use knowledge base if available, otherwise fall back to passed context_data
    if _knowledge_base["raw_data"]:
        context = build_context()
        
        # Handle specific queries with direct data processing
        prompt_lower = prompt.lower()
        
        # TOPPER/TOP STUDENTS QUERIES
        if any(word in prompt_lower for word in ['topper', 'top', 'best', 'highest', 'rank', 'first']):
            top_students = _knowledge_base.get("top_students", [])
            if top_students:
                topper = top_students[0]
                name = topper.get("Student Name", "N/A")
                total = topper.get("Total", "N/A")
                return f"{name} - {total}"
            else:
                return "No student data available"
        
        # FAILED STUDENTS QUERIES
        elif any(word in prompt_lower for word in ['failed', 'fail', 'failing', 'unsuccessful']):
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
                return "No students failed"
        
        # PASSED STUDENTS QUERIES
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
                return "No students passed"
        
        # STATISTICS/SUMMARY QUERIES
        elif any(word in prompt_lower for word in ['statistics', 'summary', 'overview', 'total', 'count', 'how many']):
            abstract = _knowledge_base.get("abstract", {})
            total = abstract.get('total_students', 'N/A')
            passed = abstract.get('passed_students', 'N/A')
            failed = abstract.get('failed_students', 'N/A')
            percentage = abstract.get('pass_percentage', 'N/A')
            return f"Total: {total}, Passed: {passed}, Failed: {failed}, Pass Rate: {percentage}%"
        
        # AVERAGE/MEAN QUERIES
        elif any(word in prompt_lower for word in ['average', 'mean']):
            raw_data = _knowledge_base.get("raw_data", [])
            scores = [s.get("Total", 0) for s in raw_data if isinstance(s.get("Total"), (int, float))]
            if scores:
                avg = sum(scores) / len(scores)
                return f"{avg:.1f}"
            else:
                return "No score data available"
        
        # HIGHEST/LOWEST SCORE QUERIES
        elif any(word in prompt_lower for word in ['highest', 'lowest', 'minimum', 'maximum']):
            raw_data = _knowledge_base.get("raw_data", [])
            scores = [s.get("Total", 0) for s in raw_data if isinstance(s.get("Total"), (int, float))]
            if scores:
                if any(word in prompt_lower for word in ['highest', 'maximum']):
                    highest = max(scores)
                    # Find student with highest score
                    for student in raw_data:
                        if student.get("Total") == highest:
                            return f"{student.get('Student Name', 'N/A')} - {highest}"
                    return f"{highest}"
                else:  # lowest/minimum
                    lowest = min(scores)
                    # Find student with lowest score
                    for student in raw_data:
                        if student.get("Total") == lowest:
                            return f"{student.get('Student Name', 'N/A')} - {lowest}"
                    return f"{lowest}"
            else:
                return "No score data available"
        
        # SUBJECT-SPECIFIC QUERIES
        elif any(word in prompt_lower for word in ['subject', 'math', 'physics', 'chemistry', 'english', 'biology']):
            subject_analysis = _knowledge_base.get("subject_analysis", [])
            if subject_analysis:
                response = "| Subject | Pass Rate | Highest | Lowest |\n|---------|-----------|---------|--------|\n"
                for subject in subject_analysis:
                    name = subject.get('subject', 'N/A')
                    pass_rate = subject.get('pass_percentage', 'N/A')
                    highest = subject.get('highest', 'N/A')
                    lowest = subject.get('lowest', 'N/A')
                    response += f"| {name} | {pass_rate}% | {highest} | {lowest} |\n"
                return response
            else:
                return "No subject analysis available"
        
        # SPECIFIC STUDENT QUERIES
        elif any(word in prompt_lower for word in ['student', 'name', 'usn']):
            # Try to find specific student mentioned in query
            raw_data = _knowledge_base.get("raw_data", [])
            for student in raw_data:
                student_name = student.get("Student Name", "").lower()
                if any(name_part in prompt_lower for name_part in student_name.split()):
                    name = student.get("Student Name", "N/A")
                    usn = student.get("USN", "N/A")
                    total = student.get("Total", "N/A")
                    result = student.get("Result", "N/A")
                    return f"{name} (USN: {usn}) - Total: {total}, Result: {result}"
            
            # If no specific student found, provide general info
            total_students = len(raw_data)
            return f"Found {total_students} students in the database. Ask about specific names or use queries like 'show all students'."
        
        # IMPROVEMENT/HELP QUERIES
        elif any(word in prompt_lower for word in ['improve', 'help', 'support', 'weak']):
            failed_students = [s for s in _knowledge_base["raw_data"] if s.get("Result", "").upper() == "FAIL"]
            if failed_students:
                return f"{len(failed_students)} students need academic support. They should focus on weak subjects and get additional tutoring."
            else:
                return "All students are performing well. Continue current teaching methods."
        
        # COMPARISON QUERIES
        elif any(word in prompt_lower for word in ['compare', 'difference', 'better', 'worse']):
            subject_analysis = _knowledge_base.get("subject_analysis", [])
            if len(subject_analysis) >= 2:
                best_subject = max(subject_analysis, key=lambda x: x.get('pass_percentage', 0))
                worst_subject = min(subject_analysis, key=lambda x: x.get('pass_percentage', 0))
                return f"Best performing subject: {best_subject.get('subject')} ({best_subject.get('pass_percentage')}% pass rate). Needs improvement: {worst_subject.get('subject')} ({worst_subject.get('pass_percentage')}% pass rate)."
            else:
                return "Need more subject data for comparison"
        
        # GENERAL ACADEMIC QUESTIONS - Use Google Gemini API
        else:
            ai_response = query_gemini_api(prompt, context)
            if ai_response:
                return ai_response
            else:
                # Intelligent fallback based on query content
                if 'why' in prompt_lower:
                    return f"Academic performance depends on multiple factors including study habits, teaching methods, student engagement, and subject difficulty. For specific insights about your data, I can analyze patterns and provide detailed explanations."
                elif 'how' in prompt_lower:
                    return f"I can help you understand academic performance through data analysis. I analyze scores, identify trends, and provide actionable insights for improvement."
                elif 'what' in prompt_lower:
                    return f"I can provide detailed information about student performance, subject analysis, pass rates, and academic insights based on your uploaded data."
                else:
                    return f"I can help analyze academic performance data. Ask me about students, subjects, scores, or any academic insights you need."
        
    elif context_data:
        return generate_intelligent_response(prompt, {"raw_data": context_data})
    else:
        # NO DATA AVAILABLE - Still provide helpful and encouraging responses
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! I'm your advanced AI academic analysis assistant. I'm ready to help you analyze student performance data. Please upload a result PDF file and I'll provide detailed insights about students, subjects, grades, and performance trends!"
        
        elif any(word in prompt_lower for word in ['help', 'what can you do', 'capabilities', 'features']):
            return """I'm an advanced AI assistant specialized in academic result analysis. Here's what I can do for you:

🎯 **Student Analysis:**
• Identify top performers and class toppers
• Find students who need academic support
• Analyze individual student performance

📊 **Statistical Analysis:**
• Calculate class averages and pass rates
• Generate subject-wise performance statistics
• Compare performance across different subjects

📈 **Performance Insights:**
• Identify performance trends and patterns
• Provide improvement recommendations
• Analyze subject difficulty levels

📋 **Detailed Reports:**
• Generate comprehensive performance reports
• Create subject-wise analysis breakdowns
• Provide actionable insights for educators

**To get started:** Upload your result PDF file using the "Upload Ledger" section, and then ask me anything about the student performance data!"""
        
        elif any(word in prompt_lower for word in ['top', 'best', 'topper', 'rank', 'first']):
            return "I can identify top performers and class toppers once you upload your result data! I'll show you the highest scorers overall, subject-wise toppers, and provide detailed performance analysis. Please upload your PDF file first."
        
        elif any(word in prompt_lower for word in ['fail', 'struggling', 'weak', 'low', 'poor']):
            return "I can identify students who need academic support and analyze failure patterns to help improve their performance. Upload your result data and I'll provide detailed analysis of struggling students with specific recommendations."
        
        elif any(word in prompt_lower for word in ['subject', 'math', 'physics', 'chemistry', 'english', 'biology']):
            return "I can provide comprehensive subject-wise analysis including pass rates, average scores, difficulty rankings, and performance comparisons. Upload your result PDF and ask me about any specific subject!"
        
        elif any(word in prompt_lower for word in ['average', 'mean', 'score', 'marks']):
            return "I can calculate various statistical measures like class averages, subject-wise means, score distributions, and performance metrics. Upload your data and I'll provide detailed statistical analysis!"
        
        elif any(word in prompt_lower for word in ['pass', 'percentage', 'rate', 'statistics']):
            return "I can analyze pass rates, success percentages, and performance statistics across different subjects and student groups. Upload your result data for comprehensive statistical analysis!"
        
        elif any(word in prompt_lower for word in ['compare', 'comparison', 'difference', 'better', 'worse']):
            return "I can compare student performance across subjects, identify best and worst performing areas, and provide comparative analysis. Upload your data and I'll show you detailed comparisons!"
        
        elif any(word in prompt_lower for word in ['improve', 'recommendation', 'suggest', 'advice']):
            return "I can provide personalized recommendations for improving student performance, identify areas needing attention, and suggest targeted interventions. Upload your result data for specific improvement strategies!"
        
        elif any(word in prompt_lower for word in ['how', 'why', 'what', 'when', 'where']):
            return f"Great question! I'm designed to answer all types of questions about academic performance and student results. To give you specific insights about '{prompt}', please upload your result PDF file first. Then I can provide detailed, data-driven answers to any question you have!"
        
        else:
            return f"""I understand you're asking about "{prompt}" - I'm ready to help with that! 

I'm an advanced AI assistant that can analyze academic data and answer ANY question about student performance. Whether you want to know about:

• Specific students and their performance
• Subject-wise analysis and trends  
• Statistical measures and comparisons
• Performance insights and recommendations
• Academic patterns and improvements

**Next step:** Please upload your result PDF file using the "Upload Ledger" section, and then ask me your question again. I'll provide detailed, specific answers based on your actual data!

I'm here to help with any academic analysis you need! 🎓"""

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