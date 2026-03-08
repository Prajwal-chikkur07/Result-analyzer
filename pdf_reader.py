import pdfplumber
import pandas as pd
import re

def extract_data_from_pdf(pdf_path):
    """
    Extracts table data from the PDF result ledger.
    Handles both grid-based tables and the new block-based textual formats.
    """
    all_data_tables = []
    text_based_students = []
    
    with pdfplumber.open(pdf_path) as pdf:
        # We will iterate through pages and try both Text Extraction and Table Extraction
        for page in pdf.pages:
            # 1. Text-based block extraction (for `sample data.pdf` style)
            text = page.extract_text()
            if text:
                curr_student = None
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    
                    usn_match = re.search(r'USN:\s*([A-Za-z0-9]+)', line)
                    if usn_match:
                        if curr_student: 
                            text_based_students.append(curr_student)
                        curr_student = {
                            "Student Name": "Unknown", 
                            "USN": usn_match.group(1), 
                            "Subjects": {}, 
                            "Total": 0, 
                            "Result": "FAIL"
                        }
                    
                    if curr_student:
                        name_match = re.search(r'Student Name:\s*([A-Za-z\s]+?)(?=\s+\d+[A-Z]|$)', line)
                        if name_match:
                            curr_student["Student Name"] = name_match.group(1).strip()
                        
                        res_match = re.search(r'Result:\s*(PASS|FAIL)', line, re.IGNORECASE)
                        if res_match:
                            curr_student["Result"] = res_match.group(1).upper()
                            
                        # Look for Marks card Total 116 + 352 0 650 468 24 186.00
                        # Captures the 5th number sequence after "Total" which is the grand total
                        total_match = re.search(r'Marks card Total.*?=\s*\d+|Marks card Total.*?\d+\s+\+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+', line, re.IGNORECASE)
                        if total_match:
                            try:
                                curr_student["Total"] = float(total_match.group(1) if total_match.group(1) else re.search(r'(\d+)', total_match.group(0)).group(1))
                            except: pass

                        # Look for subjects: 24CMLGEN3L Basic English 19 + 12 0 100 31 3 F Fail
                        # Pattern: code(8+ chars) name numbers + numbers numbers max_marks total_marks ...
                        sub_match = re.search(r'([A-Za-z0-9]{8,12})\s+(.*?)\s+(\d+)\s*\+\s*(\d+)\s+\d+\s+(100|50)\s+(\d+)\s+', line)
                        if sub_match:
                            sub_code = sub_match.group(1)
                            sub_name = sub_match.group(2).strip()
                            # Clean (cid:xxx) garage encoding issues
                            sub_name = re.sub(r'\(cid:\d+\)', '', sub_name).strip()
                            # Ensure sub_name resolves to something
                            if not sub_name or len(sub_name) < 3: 
                                sub_name = sub_code
                            
                            sub_total = float(sub_match.group(6))
                            curr_student["Subjects"][sub_name] = sub_total

                if curr_student:
                    text_based_students.append(curr_student)

            # 2. Table-based Grid Extraction (for `sample_ledger.pdf` style)
            tables = page.extract_tables()
            for table in tables:
                if not table: continue
                df = pd.DataFrame(table).dropna(how='all').reset_index(drop=True)
                if len(df) < 2: continue

                header_idx = 0
                for i, row in df.iterrows():
                    row_str = " ".join([str(x) for x in row if x]).lower()
                    if any(key in row_str for key in ['name', 'usn', 'id', 'total', 'result']):
                        header_idx = i
                        break
                
                headers = df.iloc[header_idx].tolist()
                headers = [str(h).replace('\n', ' ').strip() if h else f"Col_{i}" for i, h in enumerate(headers)]
                data_rows = df.iloc[header_idx + 1:]
                
                for _, row in data_rows.iterrows():
                    row_dict = {}
                    for i, val in enumerate(row):
                        if i < len(headers):
                            row_dict[headers[i]] = str(val).replace('\n', ' ').strip() if val else ""
                    if any(row_dict.values()):
                        all_data_tables.append(row_dict)

    # Note: text_based format seems very reliable for "sample data.pdf". 
    # If we found at least one student via text-based format (with subjects), we prefer it.
    if text_based_students and len(text_based_students) > 0 and len(text_based_students[0].get("Subjects", {})) > 0:
        return text_based_students

    # Fallback to Grid table mapping
    return process_extracted_headers(all_data_tables)


def process_extracted_headers(data):
    """
    Normalizes different header names to a standard format for grid-based tables.
    """
    if not data:
        return []
    normalized_data = []
    mapping = {
        'name': ['student name', 'name', 'candidate name'],
        'usn': ['usn', 'id', 'roll no', 'student id', 'reg no'],
        'total': ['total', 'grand total', 'sum'],
        'result': ['result', 'status', 'remarks', 'pass/fail']
    }

    for entry in data:
        new_entry = {
            "Student Name": "Unknown", "USN": "Unknown", "Subjects": {}, "Total": 0, "Result": "FAIL"
        }
        for key, value in entry.items():
            k_lower = key.lower()
            if any(m in k_lower for m in mapping['name']):
                new_entry["Student Name"] = value
            elif any(m in k_lower for m in mapping['usn']):
                new_entry["USN"] = value
            elif any(m in k_lower for m in mapping['total']):
                try: new_entry["Total"] = float(re.sub(r'[^\d.]', '', value))
                except: new_entry["Total"] = 0
            elif any(m in k_lower for m in mapping['result']):
                val_upper = str(value).upper()
                if "PASS" in val_upper or "P" == val_upper:
                    new_entry["Result"] = "PASS"
                else: new_entry["Result"] = "FAIL"
            else:
                try:
                    score = float(re.sub(r'[^\d.]', '', value))
                    if score <= 100: new_entry["Subjects"][key] = score
                except: pass
        normalized_data.append(new_entry)
        
    return normalized_data
