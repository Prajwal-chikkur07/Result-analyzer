import requests
import json
import os
import re

# Google Gemini API configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
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
    "subject_columns": [],
    "pdf_filename": ""
}


def query_gemini_api(prompt, context):
    """Query Google Gemini API for AI responses"""
    try:
        system_prompt = f"""You are an advanced AI assistant for student academic result analysis. You have access to actual student data and must provide precise, data-driven answers.

Data Context:
{context}

Instructions:
- Always use ACTUAL data from the context above — never make up numbers
- Provide DETAILED, comprehensive responses with analysis and insights
- Start with a brief summary paragraph, then show data in markdown tables, then add observations and recommendations
- Include specific student names, USNs, and marks when relevant
- If asked about a subject, match it against the available subjects (partial matches are OK)
- For recommendations, base them on actual performance data and provide actionable suggestions
- Add observations like trends, patterns, comparisons, and areas of concern
- Use markdown formatting: **bold** for emphasis, tables for data"""

        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\nQuestion: {prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
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

    # Include ALL student data for precise queries
    context += f"\n--- ALL STUDENTS ({len(raw)} records) ---\n"
    for s in raw:
        subjects_str = ", ".join(
            f"{sub}={s.get(sub, 'N/A')}" for sub in sub_cols if s.get(sub) is not None
        )
        context += (f"  {s.get('Student Name','?')} | USN: {s.get('USN','?')} | "
                    f"{subjects_str} | Total: {s.get('Total',0)} | {s.get('Result','?')}\n")

    return context


def _find_matching_subject(query_text):
    """Find the best matching subject from loaded data based on query text."""
    query_lower = query_text.lower()
    sub_cols = _knowledge_base.get("subject_columns", [])

    # Exact match first
    for sub in sub_cols:
        if sub.lower() in query_lower:
            return sub

    # Partial word match
    query_words = set(query_lower.split())
    best_match = None
    best_score = 0
    for sub in sub_cols:
        sub_words = set(sub.lower().split())
        overlap = len(query_words & sub_words)
        if overlap > best_score:
            best_score = overlap
            best_match = sub

    # Also try substring matching for short subject names
    if not best_match or best_score == 0:
        for sub in sub_cols:
            sub_lower = sub.lower()
            # Match if any significant word from subject appears in query
            for word in sub_lower.split():
                if len(word) > 3 and word in query_lower:
                    return sub

    return best_match if best_score > 0 else None


def _get_grade(marks):
    """Assign grade based on marks."""
    try:
        m = float(marks)
    except (ValueError, TypeError):
        return 'N/A'
    if m >= 90: return 'O (Outstanding)'
    if m >= 80: return 'A+'
    if m >= 70: return 'A'
    if m >= 60: return 'B+'
    if m >= 50: return 'B'
    if m >= 40: return 'C'
    if m >= 35: return 'P (Pass)'
    return 'F (Fail)'


def _get_grade_short(marks):
    """Short grade label."""
    try:
        m = float(marks)
    except (ValueError, TypeError):
        return 'N/A'
    if m >= 90: return 'O'
    if m >= 80: return 'A+'
    if m >= 70: return 'A'
    if m >= 60: return 'B+'
    if m >= 50: return 'B'
    if m >= 40: return 'C'
    if m >= 35: return 'P'
    return 'F'


def _extract_number(text):
    """Extract first number from text."""
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else None


def _build_student_table(students, include_subjects=False):
    """Build a markdown table from student records."""
    if not students:
        return "No matching students found."

    sub_cols = _knowledge_base.get("subject_columns", [])

    if include_subjects and sub_cols:
        # Short subject names for table header
        short_subs = []
        for s in sub_cols:
            parts = s.split()
            short = parts[-1] if len(parts) > 1 else s
            if len(short) > 12:
                short = short[:12]
            short_subs.append(short)

        header = "| # | Name | USN | " + " | ".join(short_subs) + " | Total | Result |\n"
        sep = "|---|------|-----|" + "|".join(["------"] * len(sub_cols)) + "|-------|--------|\n"
        rows = ""
        for i, s in enumerate(students, 1):
            marks = " | ".join(str(s.get(sub, '-')) for sub in sub_cols)
            rows += f"| {i} | {s.get('Student Name','?')} | {s.get('USN','?')} | {marks} | {s.get('Total',0)} | {s.get('Result','?')} |\n"
        return header + sep + rows
    else:
        header = "| # | Name | USN | Total | Result |\n"
        sep = "|---|------|-----|-------|--------|\n"
        rows = ""
        for i, s in enumerate(students, 1):
            rows += f"| {i} | {s.get('Student Name','?')} | {s.get('USN','?')} | {s.get('Total',0)} | {s.get('Result','?')} |\n"
        return header + sep + rows


def query_hf(prompt, context_data=None):
    """
    Advanced AI query function that provides intelligent, data-driven responses.
    Uses pattern matching for common queries and Gemini API for complex ones.
    """
    if not _knowledge_base["raw_data"]:
        if context_data:
            return generate_intelligent_response(prompt, {"raw_data": context_data})
        return _no_data_response(prompt)

    context = build_context()
    prompt_lower = prompt.lower().strip()
    raw = _knowledge_base["raw_data"]
    abstract = _knowledge_base["abstract"]
    subs_analysis = _knowledge_base["subject_analysis"]
    sub_cols = _knowledge_base["subject_columns"]
    top_students = _knowledge_base["top_students"]

    # Detect subject mentioned in query
    matched_subject = _find_matching_subject(prompt_lower)
    threshold = _extract_number(prompt_lower)

    # ── GRADE DISTRIBUTION ──
    if any(w in prompt_lower for w in ['grade', 'distribution', 'grading']):
        if matched_subject:
            grades = {}
            for s in raw:
                mark = s.get(matched_subject)
                if mark is not None:
                    g = _get_grade_short(mark)
                    grades[g] = grades.get(g, 0) + 1
            total = sum(grades.values())
            top_grade = max(grades, key=grades.get) if grades else 'N/A'
            response = f"**Grade Distribution Analysis — {matched_subject}**\n\n"
            response += f"Here is the detailed grade-wise breakdown for **{matched_subject}** across {total} students.\n\n"
            response += "| Grade | Count | Percentage |\n|-------|-------|------------|\n"
            for g in ['O', 'A+', 'A', 'B+', 'B', 'C', 'P', 'F']:
                if g in grades:
                    pct = round(grades[g] / total * 100, 1)
                    response += f"| {g} | {grades[g]} | {pct}% |\n"
            fail_count = grades.get('F', 0)
            high_count = sum(grades.get(g, 0) for g in ['O', 'A+', 'A'])
            response += f"\n**Key Observations:**\n"
            response += f"• Most common grade: **{top_grade}** ({grades[top_grade]} students)\n"
            response += f"• High performers (O/A+/A): **{high_count}** students ({round(high_count/total*100, 1)}%)\n"
            response += f"• Failed (F grade): **{fail_count}** students ({round(fail_count/total*100, 1)}%)\n"
            if fail_count > total * 0.3:
                response += f"• **Warning:** Over 30% failure rate in {matched_subject} — remedial action recommended.\n"
            return response
        else:
            grades = {}
            for s in raw:
                total = s.get("Total", 0)
                g = _get_grade_short(total)
                grades[g] = grades.get(g, 0) + 1
            total_count = len(raw)
            response = "**Overall Grade Distribution Analysis**\n\n"
            response += f"The following shows the grade distribution across all **{total_count} students** based on their total marks.\n\n"
            response += "| Grade | Range | Count | Percentage |\n|-------|-------|-------|------------|\n"
            grade_ranges = [
                ('O', '90-100'), ('A+', '80-89'), ('A', '70-79'),
                ('B+', '60-69'), ('B', '50-59'), ('C', '40-49'),
                ('P', '35-39'), ('F', 'Below 35')
            ]
            for g, rng in grade_ranges:
                if g in grades:
                    pct = round(grades[g] / total_count * 100, 1)
                    response += f"| {g} | {rng} | {grades[g]} | {pct}% |\n"
            high_count = sum(grades.get(g, 0) for g in ['O', 'A+', 'A'])
            mid_count = sum(grades.get(g, 0) for g in ['B+', 'B', 'C'])
            low_count = sum(grades.get(g, 0) for g in ['P', 'F'])
            response += f"\n**Analysis Summary:**\n"
            response += f"• **High achievers** (O/A+/A): {high_count} students ({round(high_count/total_count*100,1)}%)\n"
            response += f"• **Average performers** (B+/B/C): {mid_count} students ({round(mid_count/total_count*100,1)}%)\n"
            response += f"• **At-risk students** (P/F): {low_count} students ({round(low_count/total_count*100,1)}%)\n"
            if low_count > total_count * 0.25:
                response += f"\n**Recommendation:** {round(low_count/total_count*100,1)}% of students are at risk. Consider additional academic support programs, tutorial sessions, and mentoring for these students.\n"
            return response

    # ── TOPPER / TOP / BEST / RANK ──
    if any(w in prompt_lower for w in ['topper', 'best', 'rank 1', 'first rank']):
        if matched_subject:
            best = None
            best_score = -1
            for s in raw:
                mark = s.get(matched_subject)
                if mark is not None:
                    try:
                        val = float(mark)
                        if val > best_score:
                            best_score = val
                            best = s
                    except (ValueError, TypeError):
                        pass
            if best:
                sa = next((s for s in subs_analysis if s['subject'] == matched_subject), {})
                response = f"**{matched_subject} — Topper Analysis**\n\n"
                response += f"The highest scorer in **{matched_subject}** is:\n\n"
                response += f"| Field | Details |\n|-------|--------|\n"
                response += f"| Student Name | **{best.get('Student Name','?')}** |\n"
                response += f"| USN | {best.get('USN','?')} |\n"
                response += f"| {matched_subject} Marks | **{best_score}** |\n"
                response += f"| Overall Total | {best.get('Total',0)} |\n"
                response += f"| Overall Result | {best.get('Result','?')} |\n"
                response += f"\n**Subject Context:** {matched_subject} has a class average of {sa.get('average','N/A')} marks, "
                response += f"with {sa.get('passed',0)} students passing out of {sa.get('appeared',0)} ({sa.get('pass_percentage',0)}% pass rate). "
                response += f"The lowest score in this subject was {sa.get('lowest','N/A')} marks.\n"
                return response
            return f"No data found for {matched_subject}"
        elif top_students:
            topper = top_students[0]
            scores = [s.get("Total", 0) for s in raw if isinstance(s.get("Total"), (int, float))]
            avg = round(sum(scores) / len(scores), 1) if scores else 0
            response = f"**Class Topper Analysis**\n\n"
            response += f"The overall class topper based on total marks is:\n\n"
            response += f"| Field | Details |\n|-------|--------|\n"
            response += f"| Student Name | **{topper.get('Student Name','?')}** |\n"
            response += f"| USN | {topper.get('USN','?')} |\n"
            response += f"| Total Marks | **{topper.get('Total',0)}** |\n"
            response += f"| Result | {topper.get('Result','?')} |\n"
            response += f"\n**Class Context:** Out of {len(raw)} students, the class average is {avg} marks. "
            response += f"{abstract.get('passed_students',0)} students passed ({abstract.get('pass_percentage',0)}% pass rate). "
            if len(top_students) > 1:
                response += f"The runner-up is {top_students[1].get('Student Name','?')} with {top_students[1].get('Total',0)} marks.\n"
            return response
        return "No student data available"

    # ── TOP N STUDENTS ──
    top_n = re.search(r'top\s*(\d+)', prompt_lower)
    if top_n or (any(w in prompt_lower for w in ['top', 'highest']) and not any(w in prompt_lower for w in ['fail', 'low', 'weak'])):
        n = int(top_n.group(1)) if top_n else 5
        if matched_subject:
            sorted_by_sub = sorted(raw, key=lambda s: float(s.get(matched_subject, 0)) if str(s.get(matched_subject, '')).replace('.','').isdigit() else 0, reverse=True)
            top_list = sorted_by_sub[:n]
            sa = next((s for s in subs_analysis if s['subject'] == matched_subject), {})
            response = f"**Top {n} Performers in {matched_subject}**\n\n"
            response += f"Showing the {n} highest scorers in **{matched_subject}** (class average: {sa.get('average','N/A')}, pass rate: {sa.get('pass_percentage',0)}%).\n\n"
            response += f"| Rank | Name | USN | {matched_subject} | Result |\n|------|------|-----|------|--------|\n"
            for i, s in enumerate(top_list, 1):
                response += f"| {i} | {s.get('Student Name','?')} | {s.get('USN','?')} | {s.get(matched_subject,'-')} | {s.get('Result','?')} |\n"
            response += f"\n**Insight:** The gap between rank 1 ({top_list[0].get(matched_subject,0)}) and rank {n} ({top_list[-1].get(matched_subject,0)}) is {float(top_list[0].get(matched_subject,0)) - float(top_list[-1].get(matched_subject,0)):.0f} marks.\n"
            return response
        else:
            sorted_all = sorted(raw, key=lambda s: float(s.get('Total', 0)) if str(s.get('Total','')).replace('.','').isdigit() else 0, reverse=True)
            top_list = sorted_all[:n]
            scores = [s.get("Total", 0) for s in raw if isinstance(s.get("Total"), (int, float))]
            avg = round(sum(scores) / len(scores), 1) if scores else 0
            response = f"**Top {n} Students — Overall Performance**\n\n"
            response += f"Showing the {n} highest-scoring students out of {len(raw)} total (class average: {avg}).\n\n"
            response += _build_student_table(top_list)
            response += f"\n**Insight:** These top {n} students scored well above the class average of {avg} marks. "
            response += f"The topper ({top_list[0].get('Student Name','?')}) scored {top_list[0].get('Total',0)} marks, "
            response += f"which is {float(top_list[0].get('Total',0)) - avg:.1f} marks above the class average.\n"
            return response

    # ── FAILED STUDENTS ──
    if any(w in prompt_lower for w in ['failed', 'fail', 'failing', 'unsuccessful', 'not passed']):
        if matched_subject:
            failed = [s for s in raw if s.get(matched_subject) is not None
                      and str(s.get(matched_subject, '')).replace('.','').isdigit()
                      and float(s[matched_subject]) < 35]
            if failed:
                sa = next((s for s in subs_analysis if s['subject'] == matched_subject), {})
                fail_scores = [float(s.get(matched_subject, 0)) for s in failed]
                avg_fail = round(sum(fail_scores) / len(fail_scores), 1) if fail_scores else 0
                response = f"**Students Failed in {matched_subject} — Detailed Report**\n\n"
                response += f"**{len(failed)} out of {sa.get('appeared', len(raw))} students** scored below the passing mark of 35 in {matched_subject} "
                response += f"(failure rate: {sa.get('failed',0)}/{sa.get('appeared',0)} = {round(100 - sa.get('pass_percentage',100), 1)}%).\n\n"
                response += f"| # | Name | USN | {matched_subject} | Gap to Pass | Total | Result |\n|---|------|-----|------|------------|-------|--------|\n"
                for i, s in enumerate(failed, 1):
                    mark = float(s.get(matched_subject, 0))
                    gap = 35 - mark
                    response += f"| {i} | {s.get('Student Name','?')} | {s.get('USN','?')} | {s.get(matched_subject,'-')} | {gap:.0f} marks | {s.get('Total',0)} | {s.get('Result','?')} |\n"
                response += f"\n**Analysis:**\n"
                response += f"• Average score among failed students: **{avg_fail}** marks\n"
                response += f"• Subject pass rate: **{sa.get('pass_percentage',0)}%** (class average: {sa.get('average','N/A')})\n"
                borderline = [s for s in failed if float(s.get(matched_subject, 0)) >= 25]
                if borderline:
                    response += f"• **{len(borderline)} students are borderline** (scored 25-34) and are close to passing with minimal additional effort\n"
                response += f"\n**Recommendation:** These students need focused remedial sessions in {matched_subject}. Consider assigning peer tutors and additional practice material.\n"
                return response
            return f"Excellent! No students failed in **{matched_subject}**. All students scored 35 or above, achieving a 100% pass rate in this subject."
        else:
            failed = [s for s in raw if s.get("Result", "").upper() == "FAIL"]
            if failed:
                fail_scores = [s.get("Total", 0) for s in failed if isinstance(s.get("Total"), (int, float))]
                avg_fail = round(sum(fail_scores) / len(fail_scores), 1) if fail_scores else 0
                response = f"**Failed Students — Comprehensive Report**\n\n"
                response += f"**{len(failed)} out of {len(raw)} students** have failed the examination "
                response += f"(failure rate: {round(len(failed)/len(raw)*100, 1)}%).\n\n"
                response += _build_student_table(failed)
                # Find which subjects caused most failures
                sub_fail_counts = {}
                for s in failed:
                    for sub in sub_cols:
                        mark = s.get(sub)
                        if mark is not None and str(mark).replace('.','').isdigit() and float(mark) < 35:
                            sub_fail_counts[sub] = sub_fail_counts.get(sub, 0) + 1
                response += f"\n**Analysis:**\n"
                response += f"• Average total marks among failed students: **{avg_fail}**\n"
                response += f"• Overall pass rate: **{abstract.get('pass_percentage',0)}%**\n"
                if sub_fail_counts:
                    worst_sub = max(sub_fail_counts, key=sub_fail_counts.get)
                    response += f"• Most common subject causing failure: **{worst_sub}** ({sub_fail_counts[worst_sub]} failures)\n"
                response += f"\n**Recommendation:** Focus remedial classes on the weakest subjects. Assign mentors for at-risk students and schedule additional tutorials before the next examination.\n"
                return response
            return "Excellent result! **No students failed** — the class achieved a 100% pass rate. All students met the minimum passing criteria across all subjects."

    # ── PASSED STUDENTS ──
    if any(w in prompt_lower for w in ['passed', 'pass students', 'passing', 'successful', 'who passed']):
        if matched_subject:
            passed = [s for s in raw if s.get(matched_subject) is not None
                      and str(s.get(matched_subject, '')).replace('.','').isdigit()
                      and float(s[matched_subject]) >= 35]
            if passed:
                sa = next((s for s in subs_analysis if s['subject'] == matched_subject), {})
                pass_scores = [float(s.get(matched_subject, 0)) for s in passed]
                avg_pass = round(sum(pass_scores) / len(pass_scores), 1) if pass_scores else 0
                response = f"**Students Passed in {matched_subject} — Detailed Report**\n\n"
                response += f"**{len(passed)} out of {sa.get('appeared', len(raw))} students** scored 35 or above in {matched_subject} "
                response += f"(pass rate: {sa.get('pass_percentage',0)}%).\n\n"
                response += f"| # | Name | USN | {matched_subject} | Grade | Result |\n|---|------|-----|------|-------|--------|\n"
                for i, s in enumerate(passed, 1):
                    grade = _get_grade_short(s.get(matched_subject, 0))
                    response += f"| {i} | {s.get('Student Name','?')} | {s.get('USN','?')} | {s.get(matched_subject,'-')} | {grade} | {s.get('Result','?')} |\n"
                response += f"\n**Summary:** Average score among passed students: **{avg_pass}** marks. "
                response += f"Highest: **{sa.get('highest','N/A')}**, Lowest passing: **{min(pass_scores):.0f}**.\n"
                return response
            return f"Unfortunately, no students passed in **{matched_subject}**. All students scored below 35. Immediate remedial action is recommended."
        else:
            passed = [s for s in raw if s.get("Result", "").upper() == "PASS"]
            if passed:
                pass_scores = [s.get("Total", 0) for s in passed if isinstance(s.get("Total"), (int, float))]
                avg_pass = round(sum(pass_scores) / len(pass_scores), 1) if pass_scores else 0
                response = f"**Passed Students — Comprehensive Report**\n\n"
                response += f"**{len(passed)} out of {len(raw)} students** passed the examination "
                response += f"(pass rate: {abstract.get('pass_percentage',0)}%).\n\n"
                response += _build_student_table(passed)
                response += f"\n**Summary:** Average total among passed students: **{avg_pass}** marks. "
                response += f"These students met the passing criteria (35+) in all subjects.\n"
                return response
            return "Unfortunately, no students passed the examination. Immediate academic intervention is recommended."

    # ── ABOVE / BELOW THRESHOLD ──
    if any(w in prompt_lower for w in ['above', 'more than', 'greater than', 'scoring above', 'scored above', 'over']):
        if threshold is not None:
            if matched_subject:
                filtered = [s for s in raw if s.get(matched_subject) is not None
                            and str(s.get(matched_subject, '')).replace('.','').isdigit()
                            and float(s[matched_subject]) > threshold]
                if filtered:
                    response = f"**Students Scoring Above {threshold} in {matched_subject}** ({len(filtered)} students)\n\n"
                    response += f"| # | Name | USN | {matched_subject} | Result |\n|---|------|-----|------|--------|\n"
                    for i, s in enumerate(filtered, 1):
                        response += f"| {i} | {s.get('Student Name','?')} | {s.get('USN','?')} | {s.get(matched_subject,'-')} | {s.get('Result','?')} |\n"
                    return response
                return f"No students scored above {threshold} in {matched_subject}."
            else:
                filtered = [s for s in raw if str(s.get('Total','')).replace('.','').isdigit() and float(s['Total']) > threshold]
                if filtered:
                    response = f"**Students Scoring Above {threshold} Total** ({len(filtered)} students)\n\n"
                    return response + _build_student_table(filtered)
                return f"No students scored above {threshold} total marks."

    if any(w in prompt_lower for w in ['below', 'less than', 'under', 'scoring below', 'lower than']):
        if threshold is not None:
            if matched_subject:
                filtered = [s for s in raw if s.get(matched_subject) is not None
                            and str(s.get(matched_subject, '')).replace('.','').isdigit()
                            and float(s[matched_subject]) < threshold]
                if filtered:
                    response = f"**Students Scoring Below {threshold} in {matched_subject}** ({len(filtered)} students)\n\n"
                    response += f"| # | Name | USN | {matched_subject} | Result |\n|---|------|-----|------|--------|\n"
                    for i, s in enumerate(filtered, 1):
                        response += f"| {i} | {s.get('Student Name','?')} | {s.get('USN','?')} | {s.get(matched_subject,'-')} | {s.get('Result','?')} |\n"
                    return response
                return f"No students scored below {threshold} in {matched_subject}."
            else:
                filtered = [s for s in raw if str(s.get('Total','')).replace('.','').isdigit() and float(s['Total']) < threshold]
                if filtered:
                    response = f"**Students Scoring Below {threshold} Total** ({len(filtered)} students)\n\n"
                    return response + _build_student_table(filtered)
                return f"No students scored below {threshold} total marks."

    # ── BETWEEN / RANGE QUERIES ──
    range_match = re.search(r'between\s*(\d+)\s*(?:and|to|-)\s*(\d+)', prompt_lower)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        if matched_subject:
            filtered = [s for s in raw if s.get(matched_subject) is not None
                        and str(s.get(matched_subject, '')).replace('.','').isdigit()
                        and low <= float(s[matched_subject]) <= high]
            if filtered:
                response = f"**Students Scoring {low}-{high} in {matched_subject}** ({len(filtered)} students)\n\n"
                response += f"| # | Name | USN | {matched_subject} | Result |\n|---|------|-----|------|--------|\n"
                for i, s in enumerate(filtered, 1):
                    response += f"| {i} | {s.get('Student Name','?')} | {s.get('USN','?')} | {s.get(matched_subject,'-')} | {s.get('Result','?')} |\n"
                return response
            return f"No students scored between {low}-{high} in {matched_subject}."
        else:
            filtered = [s for s in raw if str(s.get('Total','')).replace('.','').isdigit() and low <= float(s['Total']) <= high]
            if filtered:
                response = f"**Students with Total Between {low}-{high}** ({len(filtered)} students)\n\n"
                return response + _build_student_table(filtered)
            return f"No students with total between {low}-{high}."

    # ── STATISTICS / SUMMARY / HOW MANY ──
    if any(w in prompt_lower for w in ['statistics', 'summary', 'overview', 'how many', 'count']):
        total = abstract.get('total_students', 0)
        passed_count = abstract.get('passed_students', 0)
        failed_count = abstract.get('failed_students', 0)
        pct = abstract.get('pass_percentage', 0)
        scores = [s.get("Total", 0) for s in raw if isinstance(s.get("Total"), (int, float))]
        avg = round(sum(scores) / len(scores), 1) if scores else 0
        highest = max(scores) if scores else 0
        lowest = min(scores) if scores else 0

        response = f"**Comprehensive Result Summary — {_knowledge_base['pdf_filename']}**\n\n"
        response += f"Here is the complete statistical overview of the examination results for **{total} students**.\n\n"
        response += "**Overall Statistics:**\n\n"
        response += "| Metric | Value |\n|--------|-------|\n"
        response += f"| Total Students | {total} |\n"
        response += f"| Passed | {passed_count} |\n"
        response += f"| Failed | {failed_count} |\n"
        response += f"| Pass Rate | {pct}% |\n"
        response += f"| Class Average | {avg} |\n"
        response += f"| Highest Total | {highest} |\n"
        response += f"| Lowest Total | {lowest} |\n"
        if subs_analysis:
            best_sub = max(subs_analysis, key=lambda x: x['pass_percentage'])
            worst_sub = min(subs_analysis, key=lambda x: x['pass_percentage'])
            response += f"\n**Subject Performance Overview:**\n\n"
            response += "| Subject | Pass% | Average | Highest | Lowest |\n|---------|-------|---------|---------|--------|\n"
            for sa in subs_analysis:
                response += f"| {sa['subject']} | {sa['pass_percentage']}% | {sa.get('average','N/A')} | {sa['highest']} | {sa['lowest']} |\n"
            response += f"\n**Key Insights:**\n"
            response += f"• Best performing subject: **{best_sub['subject']}** ({best_sub['pass_percentage']}% pass rate)\n"
            response += f"• Subject needing attention: **{worst_sub['subject']}** ({worst_sub['pass_percentage']}% pass rate)\n"
            response += f"• Overall class performance: {'Excellent' if pct >= 80 else 'Good' if pct >= 60 else 'Average' if pct >= 40 else 'Below expectations'} ({pct}% pass rate)\n"
            if failed_count > 0:
                response += f"• **{failed_count} students** require academic support and remedial attention\n"
        return response

    # ── AVERAGE / MEAN ──
    if any(w in prompt_lower for w in ['average', 'mean', 'avg']):
        if matched_subject:
            scores = [float(s.get(matched_subject, 0)) for s in raw
                      if s.get(matched_subject) is not None and str(s.get(matched_subject, '')).replace('.','').isdigit()]
            if scores:
                avg = round(sum(scores) / len(scores), 1)
                highest = max(scores)
                lowest = min(scores)
                above_avg = len([s for s in scores if s >= avg])
                below_avg = len(scores) - above_avg
                response = f"**{matched_subject} — Average Score Analysis**\n\n"
                response += f"The class average for **{matched_subject}** is **{avg} marks** across {len(scores)} students.\n\n"
                response += "| Metric | Value |\n|--------|-------|\n"
                response += f"| Average | {avg} |\n"
                response += f"| Highest | {highest} |\n"
                response += f"| Lowest | {lowest} |\n"
                response += f"| Above Average | {above_avg} students |\n"
                response += f"| Below Average | {below_avg} students |\n"
                response += f"| Score Range | {highest - lowest:.0f} marks |\n"
                response += f"\n**Observation:** {'Wide' if (highest - lowest) > 40 else 'Moderate' if (highest - lowest) > 20 else 'Narrow'} score distribution "
                response += f"(range: {highest - lowest:.0f} marks). {above_avg} students ({round(above_avg/len(scores)*100,1)}%) scored above the class average.\n"
                return response
            return f"No numeric data available for {matched_subject}"
        else:
            scores = [s.get("Total", 0) for s in raw if isinstance(s.get("Total"), (int, float))]
            avg = round(sum(scores) / len(scores), 1) if scores else 0
            response = "**Class Average — Detailed Analysis**\n\n"
            response += f"The overall class average is **{avg} marks** across {len(raw)} students.\n\n"
            response += "**Subject-wise Average Comparison:**\n\n"
            response += "| Subject | Average | Highest | Lowest | Pass% | Performance |\n|---------|---------|---------|--------|-------|-------------|\n"
            for sa in sorted(subs_analysis, key=lambda x: x.get('average', 0), reverse=True):
                avg_val = sa.get('average', 0)
                perf = 'Excellent' if sa['pass_percentage'] >= 80 else 'Good' if sa['pass_percentage'] >= 60 else 'Needs Improvement'
                response += f"| {sa['subject']} | {avg_val} | {sa['highest']} | {sa['lowest']} | {sa['pass_percentage']}% | {perf} |\n"
            if scores:
                response += f"\n**Overall Summary:** Class average total: **{avg}** marks (Highest: {max(scores)}, Lowest: {min(scores)}). "
                above_avg = len([s for s in scores if s >= avg])
                response += f"{above_avg} out of {len(scores)} students ({round(above_avg/len(scores)*100,1)}%) scored above the class average.\n"
            return response

    # ── SUBJECT ANALYSIS / LIST SUBJECTS ──
    if any(w in prompt_lower for w in ['subject', 'subjects', 'list all subject']):
        if matched_subject and any(w in prompt_lower for w in ['detail', 'analysis', 'stats', 'about']):
            # Detailed analysis for one subject
            sa = next((s for s in subs_analysis if s['subject'] == matched_subject), None)
            if sa:
                response = f"**{matched_subject} — Detailed Analysis**\n\n"
                response += "| Metric | Value |\n|--------|-------|\n"
                response += f"| Students Appeared | {sa['appeared']} |\n"
                response += f"| Passed | {sa['passed']} |\n"
                response += f"| Failed | {sa['failed']} |\n"
                response += f"| Pass Percentage | {sa['pass_percentage']}% |\n"
                response += f"| Highest Marks | {sa['highest']} |\n"
                response += f"| Lowest Marks | {sa['lowest']} |\n"
                response += f"| Average | {sa.get('average', 'N/A')} |\n"
                return response

        response = "**Subject-wise Result Analysis**\n\n"
        response += "| Subject | Appeared | Passed | Failed | Pass% | Highest | Lowest | Avg |\n"
        response += "|---------|----------|--------|--------|-------|---------|--------|-----|\n"
        for sa in subs_analysis:
            response += (f"| {sa['subject']} | {sa['appeared']} | {sa['passed']} | "
                        f"{sa['failed']} | {sa['pass_percentage']}% | {sa['highest']} | "
                        f"{sa['lowest']} | {sa.get('average','N/A')} |\n")
        return response

    # ── PASS PERCENTAGE / RATE ──
    if any(w in prompt_lower for w in ['pass percentage', 'pass rate', 'pass%', 'percentage']):
        if matched_subject:
            sa = next((s for s in subs_analysis if s['subject'] == matched_subject), None)
            if sa:
                return f"**{matched_subject} Pass Rate:** {sa['pass_percentage']}% ({sa['passed']} out of {sa['appeared']} passed)"
        response = "**Pass Percentage by Subject**\n\n"
        response += "| Subject | Pass Rate | Passed | Failed |\n|---------|-----------|--------|--------|\n"
        for sa in sorted(subs_analysis, key=lambda x: x['pass_percentage'], reverse=True):
            response += f"| {sa['subject']} | {sa['pass_percentage']}% | {sa['passed']} | {sa['failed']} |\n"
        response += f"\n**Overall Pass Rate: {abstract.get('pass_percentage', 0)}%**"
        return response

    # ── SPECIFIC STUDENT LOOKUP ──
    if any(w in prompt_lower for w in ['student', 'name', 'usn', 'find', 'search', 'look up']):
        # Try to find specific student by name or USN
        for student in raw:
            student_name = student.get("Student Name", "").lower()
            student_usn = student.get("USN", "").lower()
            # Check if any part of student name appears in query
            name_parts = [p for p in student_name.split() if len(p) > 2]
            if any(part in prompt_lower for part in name_parts) or student_usn in prompt_lower:
                response = f"**Student Details — {student.get('Student Name','?')}**\n\n"
                response += "| Field | Value |\n|-------|-------|\n"
                response += f"| Name | {student.get('Student Name','?')} |\n"
                response += f"| USN | {student.get('USN','?')} |\n"
                for sub in sub_cols:
                    mark = student.get(sub, '-')
                    grade = _get_grade(mark) if mark != '-' else '-'
                    response += f"| {sub} | {mark} ({grade}) |\n"
                response += f"| **Total** | **{student.get('Total',0)}** |\n"
                response += f"| **Result** | **{student.get('Result','?')}** |\n"
                return response

        return f"Student not found. There are {len(raw)} students in the database. Try searching by exact name or USN."

    # ── HIGHEST / LOWEST ──
    if any(w in prompt_lower for w in ['highest', 'lowest', 'minimum', 'maximum', 'max', 'min']):
        is_highest = any(w in prompt_lower for w in ['highest', 'maximum', 'max'])
        if matched_subject:
            scores = [(s, float(s.get(matched_subject, 0))) for s in raw
                      if s.get(matched_subject) is not None and str(s.get(matched_subject,'')).replace('.','').isdigit()]
            if scores:
                target = max(scores, key=lambda x: x[1]) if is_highest else min(scores, key=lambda x: x[1])
                label = "Highest" if is_highest else "Lowest"
                return f"**{label} in {matched_subject}:** {target[0].get('Student Name','?')} — {target[1]} marks"
        else:
            scores = [(s, float(s.get('Total', 0))) for s in raw if isinstance(s.get('Total'), (int, float))]
            if scores:
                target = max(scores, key=lambda x: x[1]) if is_highest else min(scores, key=lambda x: x[1])
                label = "Highest" if is_highest else "Lowest"
                return f"**{label} Total:** {target[0].get('Student Name','?')} (USN: {target[0].get('USN','?')}) — {target[1]} marks"
        return "No score data available"

    # ── IMPROVEMENT / WEAK / AT-RISK ──
    if any(w in prompt_lower for w in ['improve', 'improvement', 'help', 'support', 'weak', 'at risk', 'struggling', 'need']):
        failed = [s for s in raw if s.get("Result", "").upper() == "FAIL"]

        # Find weakest subjects
        weak_subjects = sorted(subs_analysis, key=lambda x: x['pass_percentage'])

        response = f"**Students Needing Academic Support**\n\n"

        if failed:
            response += f"**{len(failed)} students failed** and need immediate attention:\n\n"
            response += _build_student_table(failed)

        # Subject difficulty analysis
        response += "\n**Subject Difficulty Ranking (weakest first):**\n\n"
        response += "| Subject | Pass Rate | Failed | Recommendation |\n|---------|-----------|--------|----------------|\n"
        for sa in weak_subjects:
            if sa['pass_percentage'] < 50:
                rec = "Urgent — needs remedial classes"
            elif sa['pass_percentage'] < 70:
                rec = "Moderate — additional tutorials recommended"
            elif sa['pass_percentage'] < 90:
                rec = "Good — maintain current approach"
            else:
                rec = "Excellent — no action needed"
            response += f"| {sa['subject']} | {sa['pass_percentage']}% | {sa['failed']} | {rec} |\n"

        # Students close to passing (scored 25-34 in any subject)
        borderline = []
        for s in raw:
            for sub in sub_cols:
                mark = s.get(sub)
                if mark is not None and str(mark).replace('.','').isdigit():
                    val = float(mark)
                    if 25 <= val < 35:
                        borderline.append((s.get('Student Name','?'), sub, val))

        if borderline:
            response += "\n**Borderline Students (scored 25-34, close to passing):**\n\n"
            response += "| Student | Subject | Marks | Gap to Pass |\n|---------|---------|-------|-------------|\n"
            for name, sub, marks in borderline:
                response += f"| {name} | {sub} | {marks} | {35 - marks:.0f} marks |\n"

        return response

    # ── COMPARISON QUERIES ──
    if any(w in prompt_lower for w in ['compare', 'comparison', 'difference', 'better', 'worse', 'vs', 'versus']):
        if len(subs_analysis) >= 2:
            best = max(subs_analysis, key=lambda x: x['pass_percentage'])
            worst = min(subs_analysis, key=lambda x: x['pass_percentage'])
            avg_best = best.get('average', 0)
            avg_worst = worst.get('average', 0)

            response = "**Subject Performance Comparison — Detailed Analysis**\n\n"
            response += f"Comparing performance across **{len(subs_analysis)} subjects** for {len(raw)} students.\n\n"
            response += "| Rank | Subject | Pass% | Passed | Failed | Average | Highest | Lowest |\n"
            response += "|------|---------|-------|--------|--------|---------|---------|--------|\n"
            for rank, sa in enumerate(sorted(subs_analysis, key=lambda x: x['pass_percentage'], reverse=True), 1):
                response += (f"| {rank} | {sa['subject']} | {sa['pass_percentage']}% | {sa['passed']} | "
                            f"{sa['failed']} | {sa.get('average','N/A')} | {sa['highest']} | {sa['lowest']} |\n")
            response += f"\n**Key Findings:**\n"
            response += f"• **Best performing subject:** {best['subject']} — {best['pass_percentage']}% pass rate, average {avg_best} marks\n"
            response += f"• **Most challenging subject:** {worst['subject']} — {worst['pass_percentage']}% pass rate, average {avg_worst} marks\n"
            gap = best['pass_percentage'] - worst['pass_percentage']
            response += f"• **Performance gap:** {gap} percentage points between best and worst subjects\n"
            if gap > 30:
                response += f"\n**Concern:** The {gap}% gap between {best['subject']} and {worst['subject']} is significant. This suggests {worst['subject']} may need curriculum review, additional teaching resources, or modified assessment methods.\n"
            elif gap > 15:
                response += f"\n**Observation:** Moderate variation across subjects. Consider providing supplementary study materials for {worst['subject']}.\n"
            else:
                response += f"\n**Positive:** Relatively consistent performance across all subjects, indicating balanced teaching and student effort.\n"
            return response
        return "Need at least 2 subjects for comparison"

    # ── ALL STUDENTS ──
    if any(w in prompt_lower for w in ['all students', 'show all', 'list all', 'everyone', 'full list', 'consolidated']):
        response = f"**All Students** ({len(raw)} total)\n\n"
        return response + _build_student_table(raw, include_subjects=True)

    # ── GENERAL / COMPLEX — Use Gemini API ──
    ai_response = query_gemini_api(prompt, context)
    if ai_response:
        return ai_response

    # Fallback — intelligent pattern-based response
    return _fallback_response(prompt_lower, raw, abstract, subs_analysis)


def _fallback_response(prompt_lower, raw, abstract, subs_analysis):
    """Provide intelligent fallback when Gemini API is unavailable."""
    if 'why' in prompt_lower:
        weak = sorted(subs_analysis, key=lambda x: x['pass_percentage'])
        if weak:
            return f"Based on the data, {weak[0]['subject']} has the lowest pass rate at {weak[0]['pass_percentage']}%. This could indicate higher difficulty or areas where teaching methods need reinforcement."
    elif 'how' in prompt_lower:
        return f"The class has a {abstract.get('pass_percentage', 0)}% pass rate with {abstract.get('total_students', 0)} students. Ask me specific questions like 'show failed students' or 'top 5 students' for detailed analysis."
    elif 'what' in prompt_lower:
        return f"I have data for {len(raw)} students across {len(subs_analysis)} subjects. I can provide detailed analysis on student performance, subject statistics, grade distributions, and improvement recommendations."

    return f"I have {len(raw)} student records loaded. Try asking: 'Who is the topper?', 'Show failed students', 'Grade distribution', 'Students above 80', 'Compare subjects', or 'Who needs improvement?'"


def _no_data_response(prompt):
    """Response when no data is loaded."""
    prompt_lower = prompt.lower()

    if any(w in prompt_lower for w in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return "Hello! I'm your AI academic analysis assistant. Upload a result PDF and I'll help you analyze student performance, generate insights, and answer questions about your data."

    if any(w in prompt_lower for w in ['help', 'what can you do', 'capabilities', 'features']):
        return """I'm an advanced AI agent for academic result analysis. Here's what I can do:

**Core Analysis:**
• Consolidated result sheets with all student details
• Subject-wise analysis (pass/fail rates, averages, highest/lowest)
• Result abstracts with overall statistics
• Grade distribution analysis (O, A+, A, B+, B, C, P, F)

**Intelligent Queries:**
• "Who is the topper?" — Find top performers
• "Show students failed in [Subject]" — Subject-specific filtering
• "Students scoring above 80" — Threshold-based queries
• "Grade distribution" — Grade-wise breakdown
• "Compare subjects" — Cross-subject analysis
• "Who needs improvement?" — At-risk student identification
• "Show all students" — Full consolidated sheet

**To get started:** Upload your result PDF using the Upload Ledger section!"""

    return f"I'd love to help with '{prompt}', but I need student data first. Please upload a result PDF file and I'll provide detailed, data-driven analysis."


def generate_intelligent_response(prompt, knowledge_base):
    """Generate intelligent responses based on available data."""
    prompt_lower = prompt.lower()
    raw_data = knowledge_base.get("raw_data", [])

    if not raw_data:
        return _no_data_response(prompt)

    total_students = len(raw_data)
    passed = len([s for s in raw_data if s.get("Result", "").upper() == "PASS"])
    failed = total_students - passed

    if any(w in prompt_lower for w in ['how many', 'total', 'count']):
        return f"Total: {total_students}, Passed: {passed}, Failed: {failed}, Pass Rate: {(passed/total_students*100):.1f}%"

    if any(w in prompt_lower for w in ['average', 'mean']):
        scores = [s.get("Total", 0) for s in raw_data if isinstance(s.get("Total"), (int, float))]
        if scores:
            return f"Class Average: {sum(scores)/len(scores):.1f} (Highest: {max(scores)}, Lowest: {min(scores)})"

    return f"I have {total_students} student records. Ask me about toppers, failed students, subject analysis, grade distribution, or improvement recommendations."
