import pandas as pd
import re

def generate_analysis(data):
    """
    Analyzes the extracted student data using pandas.
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

    # Flatten subjects for easier pandas processing
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

    df = pd.DataFrame(rows)

    # 1. Result Abstract
    total_students = len(df)
    passed_students = len(df[df["Result"] == "PASS"])
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
        if sub in df.columns:
            sub_series = pd.to_numeric(df[sub], errors='coerce').dropna()
            appeared = len(sub_series)
            if appeared > 0:
                passed = len(sub_series[sub_series >= 35])
                failed = appeared - passed
                pass_perc = (passed / appeared * 100)
                subject_stats.append({
                    "subject": sub,
                    "appeared": appeared,
                    "passed": passed,
                    "failed": failed,
                    "pass_percentage": round(pass_perc, 2),
                    "highest": float(sub_series.max()),
                    "lowest": float(sub_series.min()),
                    "average": round(float(sub_series.mean()), 2)
                })

    # 3. Top Students
    top_students_df = df.sort_values(by="Total", ascending=False).head(5)
    top_students = top_students_df[["Student Name", "USN", "Total", "Result"]].to_dict(orient="records")

    return {
        "abstract": abstract,
        "subject_analysis": subject_stats,
        "top_students": top_students,
        "raw_data": rows,
        "subject_columns": all_subjects_ordered   # NEW: sent to frontend for dynamic columns
    }


def query_results(df_rows, query):
    """
    Comprehensive keyword-based structured query search.
    Returns matching rows as a list of dicts.
    """
    if not df_rows:
        return []

    df = pd.DataFrame(df_rows)
    q = query.lower().strip()

    # Identify numeric thresholds in the query
    nums = [int(x) for x in re.findall(r'\d+', q)]
    threshold = nums[0] if nums else None

    # Identify subject columns in the query
    subject_cols = [c for c in df.columns if c not in ("Student Name", "USN", "Total", "Result", "None")]
    matched_subject = next((col for col in subject_cols if col.lower() in q), None)

    # ── PASS students
    if any(k in q for k in ["passed", "pass students", "who passed", "all pass"]):
        subset = df[df["Result"] == "PASS"]
        return _format(subset, subject_cols)

    # ── FAIL students
    if any(k in q for k in ["failed", "fail students", "who failed", "not passed"]):
        if matched_subject:
            col_num = pd.to_numeric(df[matched_subject], errors='coerce')
            return _format(df[col_num < 35], subject_cols)
        return _format(df[df["Result"] == "FAIL"], subject_cols)

    # ── TOPPER / HIGHEST / RANK 1
    if any(k in q for k in ["topper", "top student", "rank 1", "first", "highest marks", "highest scorer", "best student"]):
        if matched_subject:
            col_num = pd.to_numeric(df[matched_subject], errors='coerce')
            idx = col_num.idxmax()
            return _format(df.loc[[idx]], subject_cols)
        top = df.sort_values("Total", ascending=False).head(1)
        return _format(top, subject_cols)

    # ── ABOVE THRESHOLD
    if any(k in q for k in ["above", "more than", "greater than", "scoring above", "scored above"]):
        if threshold is not None:
            if matched_subject:
                col_num = pd.to_numeric(df[matched_subject], errors='coerce')
                return _format(df[col_num > threshold], subject_cols)
            return _format(df[pd.to_numeric(df["Total"], errors='coerce') > threshold], subject_cols)

    # ── BELOW THRESHOLD
    if any(k in q for k in ["below", "less than", "under", "scoring below"]):
        if threshold is not None:
            if matched_subject:
                col_num = pd.to_numeric(df[matched_subject], errors='coerce')
                return _format(df[col_num < threshold], subject_cols)
            return _format(df[pd.to_numeric(df["Total"], errors='coerce') < threshold], subject_cols)

    # ── ALL STUDENTS
    if any(k in q for k in ["all students", "show all", "list all", "everyone", "full list"]):
        return _format(df, subject_cols)

    # ── TOP N STUDENTS
    top_n_match = re.search(r'top\s*(\d+)', q)
    if top_n_match:
        n = int(top_n_match.group(1))
        return _format(df.sort_values("Total", ascending=False).head(n), subject_cols)

    # ── SUBJECT SPECIFIC: return all students with that subject's marks
    if matched_subject:
        return _format(df[["Student Name", "USN", matched_subject, "Total", "Result"]], subject_cols)

    # ── NAME / USN SEARCH
    name_usn_mask = df.apply(
        lambda row: q in str(row.get("Student Name", "")).lower() or q in str(row.get("USN", "")).lower(),
        axis=1
    )
    if name_usn_mask.any():
        return _format(df[name_usn_mask], subject_cols)

    return []


def _format(df, subject_cols):
    """Convert DataFrame rows to serializable dicts with numeric types cleaned."""
    result = []
    for _, row in df.iterrows():
        d = {}
        for col in df.columns:
            val = row[col]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                d[col] = "-"
            elif isinstance(val, float) and val == int(val):
                d[col] = int(val)
            else:
                d[col] = val
        result.append(d)
    return result
