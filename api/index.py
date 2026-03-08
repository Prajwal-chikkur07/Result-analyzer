from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import sys
import shutil
import json
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_reader import extract_data_from_pdf
from analysis import generate_analysis, query_results
from ai_agent import query_hf, update_knowledge_base
from database import (
    save_upload_data, get_latest_upload_data, get_all_uploads, 
    get_upload_data, clear_all_data, get_database_stats, delete_upload
)

app = FastAPI(title="AI Result Analysis Agent")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories in /tmp for serverless environment
UPLOAD_DIR = "/tmp/uploads"
REPORT_DIR = "/tmp/reports"
for d in [UPLOAD_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# Session state — now backed by database
session_data = {
    "records": [],
    "analysis": {},
    "filename": "",
    "upload_id": None
}

def load_latest_session():
    """Load the latest upload data into session"""
    try:
        latest_data = get_latest_upload_data()
        if latest_data:
            session_data["records"] = latest_data["students"]
            session_data["analysis"] = latest_data["analysis"]
            session_data["filename"] = latest_data["upload_info"]["original_filename"]
            session_data["upload_id"] = latest_data["upload_info"]["id"]
            
            # Update AI knowledge base
            if latest_data["analysis"]:
                update_knowledge_base(latest_data["analysis"], latest_data["upload_info"]["original_filename"])
    except Exception as e:
        print(f"Error loading session: {e}")

# Load latest session on startup
load_latest_session()

@app.get("/")
async def root():
    return {"message": "AI Result Analysis Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is working"}

@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract student records, run analytics, and update the AI knowledge base"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract
    try:
        records = extract_data_from_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

    if not records:
        raise HTTPException(status_code=422, detail="No student records found in PDF. Check table format.")

    # Analyze
    try:
        analysis = generate_analysis(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Store in session
    session_data["records"] = records
    session_data["analysis"] = analysis
    session_data["filename"] = file.filename

    # Update AI knowledge base
    update_knowledge_base(analysis, file.filename)

    return {
        "status": "ok",
        "filename": file.filename,
        "students_loaded": len(records),
        **analysis
    }

@app.get("/ai-query")
async def ai_search_results(q: str):
    """AI-powered query endpoint"""
    try:
        # Check if we have data loaded
        if not session_data.get("analysis") or "raw_data" not in session_data["analysis"]:
            # Try to load latest session if no data
            load_latest_session()
            
            # If still no data, return helpful message
            if not session_data.get("analysis") or "raw_data" not in session_data["analysis"]:
                return {
                    "query": q, 
                    "response": "No data loaded. Upload a PDF first."
                }
        
        # Use the AI agent to handle all queries
        response = query_hf(q)
        return {"query": q, "response": response}
            
    except Exception as e:
        return {
            "query": q, 
            "response": "Please try rephrasing your question."
        }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "Uploaded successfully"}

@app.get("/extract")
async def extract_results(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        data = extract_data_from_pdf(file_path)
        session_data["records"] = data
        session_data["filename"] = filename
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/analysis")
async def get_analysis():
    if not session_data["records"]:
        raise HTTPException(status_code=400, detail="No data extracted yet")
    try:
        analysis = generate_analysis(session_data["records"])
        session_data["analysis"] = analysis
        update_knowledge_base(analysis, session_data.get("filename", ""))
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/query")
async def search_results(q: str):
    if not session_data.get("analysis") or "raw_data" not in session_data["analysis"]:
        raise HTTPException(status_code=400, detail="No analysis data available to query")
    try:
        results = query_results(session_data["analysis"]["raw_data"], q)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/uploads")
async def get_uploads():
    """Get all uploads from database"""
    try:
        uploads = get_all_uploads()
        return uploads
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get uploads: {str(e)}")

@app.delete("/uploads/{upload_id}")
async def delete_upload_endpoint(upload_id: int):
    """Delete an upload from database"""
    try:
        delete_upload(upload_id)
        return {"status": "success", "message": "Upload deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete upload: {str(e)}")

@app.get("/settings")
async def get_settings():
    """Get database settings and statistics"""
    try:
        stats = get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")

@app.post("/database/clear")
async def clear_database():
    """Clear all data from database"""
    try:
        clear_all_data()
        # Reset session data
        session_data["records"] = []
        session_data["analysis"] = {}
        session_data["filename"] = ""
        session_data["upload_id"] = None
        return {"status": "success", "message": "All data cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

@app.get("/current-data")
async def get_current_data():
    """Get current session data"""
    try:
        if session_data.get("analysis") and "raw_data" in session_data["analysis"]:
            return {
                "hasData": True,
                "filename": session_data.get("filename", ""),
                "students_count": len(session_data.get("records", [])),
                **session_data["analysis"]
            }
        else:
            # Try to load latest data
            load_latest_session()
            if session_data.get("analysis") and "raw_data" in session_data["analysis"]:
                return {
                    "hasData": True,
                    "filename": session_data.get("filename", ""),
                    "students_count": len(session_data.get("records", [])),
                    **session_data["analysis"]
                }
            else:
                return {"hasData": False}
    except Exception as e:
        return {"hasData": False, "error": str(e)}

# Vercel handler
handler = app