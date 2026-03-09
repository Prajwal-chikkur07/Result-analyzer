import re

def generate_analysis(data):
    """
    Analyzes the extracted student data using native Python.
    Returns: abstract, subject_analysis, top_students, raw_data, subject_columns
    """
    if not data:
        return {
            "abstract": {},
            "subject_analysis": [],
            "top_students": [],
            "raw_data": [],
            "subject_columns": []
        }

    # Flatten subjects for easier processing
    rows = []
    all_subjects_ordered = []  # preserve order
    seen_subjects = set()
    for student in data:
        for sub in student["Subjects"].keys():
            if sub not in seen_subjects:
                all_subjects_ordered.append(sub)
                seen_subjects.add(sub)

    for student in data:
        row = {
            "Student Name": student["Student Name"],
            "USN": student["USN"],
        }
        for sub in all_subjects_ordered:
            row[sub] = student["Subjects"].get(sub, None)
        row["Total"] = student["Total"]
        row["Result"] = student["Result"]
        rows.append(row)

    # 1. Result Abstract
    total_students = len(rows)
    passed_students = len([r for r in rows if r["Result"] == "PASS"])
    failed_students = total_students - passed_students
    pass_percentage = (passed_students / total_students * 100) if total_students > 0 else 0

    abstract = {
        "total_students": total_students,
        "passed_students": passed_students,
        "failed_students": failed_students,
        "pass_percentage": round(pass_percentage, 2)
    }

    # 2. Subject Analysis
    subject_stats = []
    for sub in all_subjects_ordered:
        # Get all numeric scores for this subject
        scores = []
        for row in rows:
            score = row.get(sub)
            if score is not None and str(score).replace('.', '').isdigit():
                scores.append(float(score))
        
        if scores:
            appeared = len(scores)
            passed = len([s for s in scores if s >= 35])
            failed = appeared - passed
            pass_perc = (passed / appeared * 100)
            subject_stats.append({
                "subject": sub,
                "appeared": appeared,
                "passed": passed,
                "failed": failed,
                "pass_percentage": round(pass_perc, 2),
                "highest": max(scores),
                "lowest": min(scores),
                "average": round(sum(scores) / len(scores), 2)
            })

    # 3. Top Students
    # Sort by total marks (descending)
    sorted_students = sorted(rows, key=lambda x: float(x["Total"]) if str(x["Total"]).replace('.', '').isdigit() else 0, reverse=True)
    top_students = []
    for student in sorted_students[:5]:
        top_students.append({
            "Student Name": student["Student Name"],
            "USN": student["USN"],
            "Total": student["Total"],
            "Result": student["Result"]
        })

    # 4. Subject-wise Student Details with Performance Reports
    subject_student_details = []
    for sub in all_subjects_ordered:
        sub_stat = next((s for s in subject_stats if s["subject"] == sub), None)
        sub_avg = sub_stat["average"] if sub_stat else 0
        sub_highest = sub_stat["highest"] if sub_stat else 0

        students_in_subject = []
        for row in rows:
            score = row.get(sub)
            if score is None:
                continue
            if not str(score).replace('.', '').isdigit():
                continue
            score_val = float(score)

            # Grade
            if score_val >= 90:
                grade = "O"
            elif score_val >= 80:
                grade = "A+"
            elif score_val >= 70:
                grade = "A"
            elif score_val >= 60:
                grade = "B+"
            elif score_val >= 50:
                grade = "B"
            elif score_val >= 40:
                grade = "C"
            elif score_val >= 35:
                grade = "P"
            else:
                grade = "F"

            # Performance relative to class
            if score_val >= sub_avg + 15:
                performance = "Excellent"
            elif score_val >= sub_avg + 5:
                performance = "Above Average"
            elif score_val >= sub_avg - 5:
                performance = "Average"
            elif score_val >= sub_avg - 15:
                performance = "Below Average"
            else:
                performance = "Needs Improvement"

            # Lagging areas and improvement suggestions
            lagging = []
            improvements = []

            if score_val < 35:
                lagging.append("Failed — needs focused revision on core concepts")
                improvements.append("Revisit fundamentals and practice previous year questions")
                improvements.append("Seek tutoring or join study groups for this subject")
            elif score_val < 50:
                lagging.append("Borderline performance — at risk of failing")
                improvements.append("Strengthen weak topics through targeted practice")
                improvements.append("Allocate more dedicated study time")
            elif score_val < 60:
                lagging.append("Below class average — significant room for growth")
                improvements.append("Focus on application-based and problem-solving questions")
                improvements.append("Review notes regularly and attempt mock tests")

            if score_val < sub_avg:
                gap = round(sub_avg - score_val, 1)
                lagging.append(f"Scoring {gap} marks below class average ({sub_avg})")
                improvements.append(f"Target at least {int(sub_avg + 5)} marks to rise above average")

            if score_val >= sub_avg and score_val < sub_highest - 10:
                improvements.append(f"Aim for top scorer range ({int(sub_highest)} marks)")

            # Check if this is the student's weakest subject
            student_scores = {}
            for other_sub in all_subjects_ordered:
                other_score = row.get(other_sub)
                if other_score is not None and str(other_score).replace('.', '').isdigit():
                    student_scores[other_sub] = float(other_score)

            if len(student_scores) > 1:
                avg_other = sum(v for k, v in student_scores.items() if k != sub) / (len(student_scores) - 1)
                if score_val < avg_other - 10:
                    lagging.append(f"Weakest subject — {round(avg_other - score_val, 1)} marks below own average ({round(avg_other, 1)})")
                    improvements.append("Prioritize this subject over stronger ones")

            if not lagging:
                lagging.append("Performing well — no major concerns")
            if not improvements:
                improvements.append("Maintain consistency and aim higher")

            students_in_subject.append({
                "name": row["Student Name"],
                "usn": row["USN"],
                "marks": score_val,
                "grade": grade,
                "performance": performance,
                "lagging": lagging,
                "improvements": improvements,
                "total": row.get("Total", 0),
                "result": row.get("Result", "")
            })

        students_in_subject.sort(key=lambda x: x["marks"], reverse=True)

        subject_student_details.append({
            "subject": sub,
            "class_average": sub_avg,
            "highest": sub_highest,
            "total_students": len(students_in_subject),
            "passed": len([s for s in students_in_subject if s["marks"] >= 35]),
            "failed": len([s for s in students_in_subject if s["marks"] < 35]),
            "students": students_in_subject
        })

    return {
        "abstract": abstract,
        "subject_analysis": subject_stats,
        "top_students": top_students,
        "raw_data": rows,
        "subject_columns": all_subjects_ordered,
        "subject_student_details": subject_student_details
    }


def query_results(rows, query):
    """
    Comprehensive keyword-based structured query search.
    Returns matching rows as a list of dicts.
    """
    if not rows:
        return []

    q = query.lower().strip()

    # Identify numeric thresholds in the query
    nums = [int(x) for x in re.findall(r'\d+', q)]
    threshold = nums[0] if nums else None

    # Identify subject columns in the query
    subject_cols = []
    if rows:
        subject_cols = [k for k in rows[0].keys() if k not in ("Student Name", "USN", "Total", "Result")]
    matched_subject = next((col for col in subject_cols if col.lower() in q), None)

    # ── PASS students
    if any(k in q for k in ["passed", "pass students", "who passed", "all pass"]):
        return [row for row in rows if row["Result"] == "PASS"]

    # ── FAIL students
    if any(k in q for k in ["failed", "fail students", "who failed", "not passed"]):
        if matched_subject:
            result = []
            for row in rows:
                score = row.get(matched_subject)
                if score is not None and str(score).replace('.', '').isdigit() and float(score) < 35:
                    result.append(row)
            return result
        return [row for row in rows if row["Result"] == "FAIL"]

    # ── TOPPER / HIGHEST / RANK 1
    if any(k in q for k in ["topper", "top student", "rank 1", "first", "highest marks", "highest scorer", "best student"]):
        if matched_subject:
            best_score = -1
            best_student = None
            for row in rows:
                score = row.get(matched_subject)
                if score is not None and str(score).replace('.', '').isdigit():
                    score_val = float(score)
                    if score_val > best_score:
                        best_score = score_val
                        best_student = row
            return [best_student] if best_student else []
        
        # Find student with highest total
        best_total = -1
        best_student = None
        for row in rows:
            total = row.get("Total")
            if total is not None and str(total).replace('.', '').isdigit():
                total_val = float(total)
                if total_val > best_total:
                    best_total = total_val
                    best_student = row
        return [best_student] if best_student else []

    # ── ABOVE THRESHOLD
    if any(k in q for k in ["above", "more than", "greater than", "scoring above", "scored above"]):
        if threshold is not None:
            result = []
            for row in rows:
                if matched_subject:
                    score = row.get(matched_subject)
                    if score is not None and str(score).replace('.', '').isdigit() and float(score) > threshold:
                        result.append(row)
                else:
                    total = row.get("Total")
                    if total is not None and str(total).replace('.', '').isdigit() and float(total) > threshold:
                        result.append(row)
            return result

    # ── BELOW THRESHOLD
    if any(k in q for k in ["below", "less than", "under", "scoring below"]):
        if threshold is not None:
            result = []
            for row in rows:
                if matched_subject:
                    score = row.get(matched_subject)
                    if score is not None and str(score).replace('.', '').isdigit() and float(score) < threshold:
                        result.append(row)
                else:
                    total = row.get("Total")
                    if total is not None and str(total).replace('.', '').isdigit() and float(total) < threshold:
                        result.append(row)
            return result

    # ── ALL STUDENTS
    if any(k in q for k in ["all students", "show all", "list all", "everyone", "full list"]):
        return rows

    # ── TOP N STUDENTS
    top_n_match = re.search(r'top\s*(\d+)', q)
    if top_n_match:
        n = int(top_n_match.group(1))
        sorted_students = sorted(rows, key=lambda x: float(x["Total"]) if str(x["Total"]).replace('.', '').isdigit() else 0, reverse=True)
        return sorted_students[:n]

    # ── SUBJECT SPECIFIC: return all students with that subject's marks
    if matched_subject:
        result = []
        for row in rows:
            filtered_row = {
                "Student Name": row["Student Name"],
                "USN": row["USN"],
                matched_subject: row.get(matched_subject, "-"),
                "Total": row["Total"],
                "Result": row["Result"]
            }
            result.append(filtered_row)
        return result

    # ── NAME / USN SEARCH
    result = []
    for row in rows:
        name = str(row.get("Student Name", "")).lower()
        usn = str(row.get("USN", "")).lower()
        if q in name or q in usn:
            result.append(row)
    
    return result
