import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DATABASE_PATH = os.path.join(os.getcwd(), "result_analyzer.db")

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create uploads table to store PDF metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            status TEXT DEFAULT 'processed'
        )
    ''')
    
    # Create students table to store individual student records
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER,
            student_name TEXT NOT NULL,
            usn TEXT,
            total_marks INTEGER,
            result TEXT,
            subject_marks TEXT,  -- JSON string of subject marks
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES uploads (id) ON DELETE CASCADE
        )
    ''')
    
    # Create analysis table to store analysis results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER,
            total_students INTEGER,
            passed_students INTEGER,
            failed_students INTEGER,
            pass_percentage REAL,
            subject_analysis TEXT,  -- JSON string
            top_students TEXT,      -- JSON string
            subject_columns TEXT,   -- JSON string
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES uploads (id) ON DELETE CASCADE
        )
    ''')
    
    # Create settings table for application settings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_upload_data(filename: str, original_filename: str, file_size: int, 
                    student_records: List[Dict], analysis_data: Dict) -> int:
    """Save uploaded PDF data and analysis to database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Insert upload record
        cursor.execute('''
            INSERT INTO uploads (filename, original_filename, file_size)
            VALUES (?, ?, ?)
        ''', (filename, original_filename, file_size))
        
        upload_id = cursor.lastrowid
        
        # Insert student records
        for student in student_records:
            # Handle both formats: nested "Subjects" dict or flattened columns
            if 'Subjects' in student and isinstance(student['Subjects'], dict):
                subject_marks = student['Subjects']
            else:
                subject_marks = {k: v for k, v in student.items()
                               if k not in ['Student Name', 'USN', 'Total', 'Result']}

            cursor.execute('''
                INSERT INTO students (upload_id, student_name, usn, total_marks, result, subject_marks)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                upload_id,
                student.get('Student Name', ''),
                student.get('USN', ''),
                student.get('Total', 0),
                student.get('Result', ''),
                json.dumps(subject_marks)
            ))
        
        # Insert analysis data
        cursor.execute('''
            INSERT INTO analysis (upload_id, total_students, passed_students, failed_students, 
                                pass_percentage, subject_analysis, top_students, subject_columns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            upload_id,
            analysis_data.get('abstract', {}).get('total_students', 0),
            analysis_data.get('abstract', {}).get('passed_students', 0),
            analysis_data.get('abstract', {}).get('failed_students', 0),
            analysis_data.get('abstract', {}).get('pass_percentage', 0),
            json.dumps(analysis_data.get('subject_analysis', [])),
            json.dumps(analysis_data.get('top_students', [])),
            json.dumps(analysis_data.get('subject_columns', []))
        ))
        
        conn.commit()
        return upload_id
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_all_uploads() -> List[Dict]:
    """Get all uploaded files with their metadata"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT u.id, u.filename, u.original_filename, u.upload_date, u.file_size,
               a.total_students, a.passed_students, a.failed_students, a.pass_percentage
        FROM uploads u
        LEFT JOIN analysis a ON u.id = a.upload_id
        ORDER BY u.upload_date DESC
    ''')
    
    uploads = []
    for row in cursor.fetchall():
        uploads.append({
            'id': row[0],
            'filename': row[1],
            'original_filename': row[2],
            'upload_date': row[3],
            'file_size': row[4],
            'total_students': row[5] or 0,
            'passed_students': row[6] or 0,
            'failed_students': row[7] or 0,
            'pass_percentage': row[8] or 0
        })
    
    conn.close()
    return uploads

def get_upload_data(upload_id: int) -> Optional[Dict]:
    """Get complete data for a specific upload"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get upload info
    cursor.execute('SELECT * FROM uploads WHERE id = ?', (upload_id,))
    upload_row = cursor.fetchone()
    if not upload_row:
        conn.close()
        return None
    
    # Get students data
    cursor.execute('SELECT * FROM students WHERE upload_id = ?', (upload_id,))
    student_rows = cursor.fetchall()
    
    # Get analysis data
    cursor.execute('SELECT * FROM analysis WHERE upload_id = ?', (upload_id,))
    analysis_row = cursor.fetchone()
    
    conn.close()
    
    # Build student records in the format expected by generate_analysis()
    students = []
    for row in student_rows:
        subject_marks = json.loads(row[6]) if row[6] else {}
        # Handle double-nested Subjects (legacy data stored as {"Subjects": {...}})
        if 'Subjects' in subject_marks and isinstance(subject_marks['Subjects'], dict):
            subject_marks = subject_marks['Subjects']
        student = {
            'Student Name': row[2],
            'USN': row[3],
            'Subjects': subject_marks,
            'Total': row[4],
            'Result': row[5]
        }
        students.append(student)
    
    # Build analysis data
    analysis = {}
    if analysis_row:
        analysis = {
            'abstract': {
                'total_students': analysis_row[2],
                'passed_students': analysis_row[3],
                'failed_students': analysis_row[4],
                'pass_percentage': analysis_row[5]
            },
            'subject_analysis': json.loads(analysis_row[6]) if analysis_row[6] else [],
            'top_students': json.loads(analysis_row[7]) if analysis_row[7] else [],
            'subject_columns': json.loads(analysis_row[8]) if analysis_row[8] else [],
            'raw_data': students
        }
    
    return {
        'upload_info': {
            'id': upload_row[0],
            'filename': upload_row[1],
            'original_filename': upload_row[2],
            'upload_date': upload_row[3],
            'file_size': upload_row[4]
        },
        'students': students,
        'analysis': analysis
    }

def get_latest_upload_data() -> Optional[Dict]:
    """Get the most recent upload data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM uploads ORDER BY upload_date DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return get_upload_data(row[0])
    return None

def clear_all_data():
    """Clear all data from the database permanently"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:
        # Delete all records from all tables
        cursor.execute('DELETE FROM analysis')
        cursor.execute('DELETE FROM students')
        cursor.execute('DELETE FROM uploads')
        cursor.execute('DELETE FROM settings')

        # Reset auto-increment counters
        try:
            cursor.execute('DELETE FROM sqlite_sequence')
        except Exception:
            pass

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

    # VACUUM must run outside a transaction
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute('VACUUM')
        conn.close()
    except Exception:
        pass

    return True

def get_database_stats() -> Dict:
    """Get database statistics"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Count records in each table
    cursor.execute('SELECT COUNT(*) FROM uploads')
    uploads_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM students')
    students_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM analysis')
    analysis_count = cursor.fetchone()[0]
    
    # Get database file size
    db_size = os.path.getsize(DATABASE_PATH) if os.path.exists(DATABASE_PATH) else 0
    
    conn.close()
    
    return {
        'uploads_count': uploads_count,
        'students_count': students_count,
        'analysis_count': analysis_count,
        'database_size_bytes': db_size,
        'database_size_mb': round(db_size / (1024 * 1024), 2)
    }

def delete_upload(upload_id: int) -> bool:
    """Delete a specific upload and all related data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('DELETE FROM analysis WHERE upload_id = ?', (upload_id,))
        cursor.execute('DELETE FROM students WHERE upload_id = ?', (upload_id,))
        cursor.execute('DELETE FROM uploads WHERE id = ?', (upload_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# Initialize database when module is imported
init_database()