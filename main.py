from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import json
from datetime import datetime
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

# Serve static files (CSS, JS, etc.)
if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
REPORT_DIR = os.path.join(os.getcwd(), "reports")
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
    """Load the latest upload data into session and update AI knowledge base"""
    try:
        latest_data = get_latest_upload_data()
        if latest_data:
            session_data["records"] = latest_data["students"]
            session_data["analysis"] = latest_data["analysis"]
            session_data["filename"] = latest_data["upload_info"]["original_filename"]
            session_data["upload_id"] = latest_data["upload_info"]["id"]
            
            # CRITICAL: Update AI knowledge base so AI can work with the data
            if latest_data["analysis"]:
                update_knowledge_base(latest_data["analysis"], latest_data["upload_info"]["original_filename"])
                print(f"AI knowledge base updated with data from {latest_data['upload_info']['original_filename']}")
                print(f"Loaded {len(latest_data['students'])} students for AI analysis")
            return True
        return False
    except Exception as e:
        print(f"Error loading latest session: {e}")
        return False

# Load latest session on startup
try:
    from database import init_database
    init_database()
    print("Database initialized successfully")
except Exception as e:
    print(f"Warning: Database initialization failed: {e}")

load_latest_session()

# ─────────────────────────────────────────────────────────────────────────────
# Frontend Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML"""
    try:
        if os.path.exists("index.html"):
            with open("index.html", "r") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content="""
            <html>
                <head><title>AI Result Analysis Agent</title></head>
                <body>
                    <h1>AI Result Analysis Agent</h1>
                    <p>Backend is running successfully!</p>
                    <p>API endpoints are available at:</p>
                    <ul>
                        <li><a href="/docs">/docs</a> - API Documentation</li>
                        <li><a href="/process">/process</a> - Upload and process PDF</li>
                        <li><a href="/ai-query?q=hello">/ai-query</a> - AI Query endpoint</li>
                    </ul>
                </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Result Analysis Agent is running"}

# ─────────────────────────────────────────────────────────────────────────────
# /process — One-shot endpoint: upload + extract + analyse in one call
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract student records, run analytics, store in database, and update the AI knowledge base
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # 1. Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Extract
    try:
        records = extract_data_from_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

    if not records:
        raise HTTPException(status_code=422, detail="No student records found in PDF. Check table format.")

    # 3. Analyze
    try:
        analysis = generate_analysis(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # 4. Store in session AND database
    session_data["records"] = records
    session_data["analysis"] = analysis
    session_data["filename"] = file.filename

    # 5. Save to database for persistence
    try:
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        upload_id = save_upload_data(file.filename, file.filename, file_size, records, analysis)
        session_data["upload_id"] = upload_id
        print(f"Data saved to database with upload_id: {upload_id}")
    except Exception as e:
        print(f"Warning: Could not save to database: {e}")

    # 6. Update AI knowledge base (CRITICAL - this makes AI work with the data)
    update_knowledge_base(analysis, file.filename)
    print(f"AI knowledge base updated with {len(records)} students from {file.filename}")

    return {
        "status": "ok",
        "filename": file.filename,
        "students_loaded": len(records),
        **analysis
    }

# ─────────────────────────────────────────────────────────────────────────────
# Legacy endpoints — kept for backward compat
# ─────────────────────────────────────────────────────────────────────────────
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

@app.get("/settings")
async def get_settings():
    """Get database statistics and settings"""
    try:
        stats = get_database_stats()
        return {
            "database_stats": stats,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting settings: {str(e)}"
        }

@app.get("/uploads")
async def get_uploads():
    """Get all uploads from database"""
    try:
        uploads = get_all_uploads()
        return {
            "uploads": uploads,
            "status": "success"
        }
    except Exception as e:
        return {
            "uploads": [],
            "status": "error",
            "message": f"Error getting uploads: {str(e)}"
        }

@app.post("/database/clear")
async def clear_database():
    """Clear all data from database"""
    try:
        clear_all_data()
        # Also clear session data
        session_data["records"] = []
        session_data["analysis"] = {}
        session_data["filename"] = ""
        session_data["upload_id"] = None
        
        # Clear AI knowledge base
        from ai_agent import _knowledge_base
        _knowledge_base["raw_data"] = []
        _knowledge_base["abstract"] = {}
        _knowledge_base["subject_analysis"] = []
        _knowledge_base["top_students"] = []
        _knowledge_base["subject_columns"] = []
        _knowledge_base["pdf_filename"] = ""
        
        return {
            "status": "success",
            "message": "All data cleared successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error clearing data: {str(e)}"
        }

@app.post("/uploads/{upload_id}/load")
async def load_upload(upload_id: int):
    """Load a specific upload by ID"""
    try:
        upload_data = get_upload_data(upload_id)
        if upload_data:
            session_data["records"] = upload_data["students"]
            session_data["analysis"] = upload_data["analysis"]
            session_data["filename"] = upload_data["upload_info"]["original_filename"]
            session_data["upload_id"] = upload_data["upload_info"]["id"]
            
            # Update AI knowledge base
            if upload_data["analysis"]:
                update_knowledge_base(upload_data["analysis"], upload_data["upload_info"]["original_filename"])
            
            return {
                "status": "success",
                "message": f"Upload {upload_id} loaded successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Upload not found"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading upload: {str(e)}"
        }

@app.delete("/uploads/{upload_id}")
async def delete_upload_endpoint(upload_id: int):
    """Delete a specific upload by ID"""
    try:
        delete_upload(upload_id)
        return {
            "status": "success",
            "message": f"Upload {upload_id} deleted successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting upload: {str(e)}"
        }

@app.get("/current-data")
async def get_current_data():
    """Get the currently loaded data for the frontend"""
    try:
        # Ensure we have the latest data loaded
        if not session_data.get("analysis"):
            load_latest_session()
        
        if session_data.get("analysis") and "raw_data" in session_data["analysis"]:
            return {
                "status": "success",
                "data": session_data["analysis"],
                "filename": session_data.get("filename", "Unknown"),
                "students_count": len(session_data["analysis"]["raw_data"])
            }
        else:
            return {
                "status": "no_data",
                "message": "No data currently loaded"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading data: {str(e)}"
        }

@app.get("/data-status")
async def get_data_status():
    """Check what data is currently loaded in the system"""
    try:
        # Try to load latest data if none in session
        if not session_data.get("analysis"):
            load_latest_session()
        
        if session_data.get("analysis") and "raw_data" in session_data["analysis"]:
            return {
                "status": "data_loaded",
                "filename": session_data.get("filename", "Unknown"),
                "students_count": len(session_data["analysis"]["raw_data"]),
                "upload_id": session_data.get("upload_id"),
                "subjects": session_data["analysis"].get("subject_columns", [])
            }
        else:
            return {
                "status": "no_data",
                "message": "No data currently loaded. Please upload a PDF file."
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking data status: {str(e)}"
        }

@app.get("/load-latest")
async def load_latest_data():
    """Load the latest upload data into the current session"""
    try:
        load_latest_session()
        if session_data.get("analysis") and "raw_data" in session_data["analysis"]:
            return {
                "status": "success",
                "message": f"Loaded data for {len(session_data['analysis']['raw_data'])} students",
                "filename": session_data.get("filename", "Unknown")
            }
        else:
            return {
                "status": "no_data",
                "message": "No previous data found in database"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading data: {str(e)}"
        }

@app.get("/ai-query")
async def ai_search_results(q: str):
    """
    ADVANCED AI Query endpoint that ALWAYS provides intelligent responses.
    This endpoint never fails to give a helpful answer and always works with uploaded data.
    """
    try:
        # Always try to ensure we have the latest data loaded
        if not session_data.get("analysis") or "raw_data" not in session_data["analysis"]:
            print("No data in session, attempting to load latest data...")
            data_loaded = load_latest_session()
            if data_loaded:
                print("Successfully loaded data from database")
            else:
                print("No data found in database")
        
        # Check if we now have data
        if session_data.get("analysis") and "raw_data" in session_data["analysis"]:
            print(f"AI processing query with {len(session_data['analysis']['raw_data'])} students")
        
        # Use the enhanced AI agent to handle ALL queries
        response = query_hf(q)
        return {"query": q, "response": response}
            
    except Exception as e:
        print(f"Error in AI query: {e}")
        # Even if there's an error, provide a helpful response
        return {
            "query": q, 
            "response": f"I'm your advanced AI assistant for academic analysis. I can help with student performance, grades, and insights. Please upload your result data and ask me anything!"
        }

# ─────────────────────────────────────────────────────────────────────────────
# /report — Generate and download a PDF report for the current analysis
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/report")
async def generate_report():
    if not session_data.get("analysis") or "raw_data" not in session_data["analysis"]:
        raise HTTPException(status_code=400, detail="No analysis data. Upload a PDF first.")

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph,
            Spacer, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        analysis = session_data["analysis"]
        ab = analysis["abstract"]
        subs = analysis["subject_analysis"]
        raw = analysis["raw_data"]
        top = analysis["top_students"]
        filename = session_data.get("filename", "result_ledger.pdf")

        report_path = os.path.join(REPORT_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        doc = SimpleDocTemplate(report_path, pagesize=A4,
                                rightMargin=1.5*cm, leftMargin=1.5*cm,
                                topMargin=1.5*cm, bottomMargin=1.5*cm)
        styles = getSampleStyleSheet()
        elements = []

        # ── Title ──────────────────────────────────────────────────────────
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=18, leading=24,
                                     alignment=TA_CENTER, spaceAfter=4)
        sub_style = ParagraphStyle("sub", parent=styles["Normal"],
                                   fontSize=10, alignment=TA_CENTER,
                                   textColor=colors.grey, spaceAfter=12)

        elements.append(Paragraph("AI Result Analysis Report", title_style))
        elements.append(Paragraph(
            f"Source: {filename} &nbsp;|&nbsp; Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            sub_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#6366f1"), spaceAfter=14))

        # ── Abstract Cards ─────────────────────────────────────────────────
        elements.append(Paragraph("Result Abstract", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        abstract_data = [
            ["Total Students", "Passed", "Failed", "Pass Percentage"],
            [ab["total_students"], ab["passed_students"], ab["failed_students"], f"{ab['pass_percentage']}%"]
        ]
        abstract_table = Table(abstract_data, colWidths=[4*cm]*4)
        abstract_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#6366f1")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 10),
            ("BACKGROUND",   (0, 1), (-1, -1), colors.HexColor("#f0f0ff")),
            ("FONTSIZE",     (0, 1), (-1, -1), 14),
            ("FONTNAME",     (0, 1), (-1, -1), "Helvetica-Bold"),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eef0ff")]),
            ("TOPPADDING",   (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ]))
        elements.append(abstract_table)
        elements.append(Spacer(1, 18))

        # ── Subject Analysis Table ─────────────────────────────────────────
        elements.append(Paragraph("Subject-wise Analysis", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        if subs:
            sub_header = ["Subject", "Appeared", "Passed", "Failed", "Pass %", "Highest", "Lowest"]
            sub_rows = [sub_header] + [
                [s["subject"], s["appeared"], s["passed"], s["failed"],
                 f"{s['pass_percentage']}%", s["highest"], s["lowest"]]
                for s in subs
            ]
            sub_table = Table(sub_rows, colWidths=[4.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm])
            sub_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("ALIGN",         (0, 0), (0, -1), "LEFT"),
                ("FONTSIZE",      (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5ff")]),
                ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d0e8")),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ]))
            elements.append(sub_table)
        elements.append(Spacer(1, 18))

        # ── Top Students ───────────────────────────────────────────────────
        if top:
            elements.append(Paragraph("Top 5 Performers", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            top_header = ["Rank", "Student Name", "USN", "Total Marks", "Result"]
            top_rows = [top_header] + [
                [i+1, s.get("Student Name","?"), s.get("USN","?"), s.get("Total",0), s.get("Result","?")]
                for i, s in enumerate(top)
            ]
            top_table = Table(top_rows, colWidths=[1.5*cm, 5.5*cm, 3.5*cm, 3*cm, 2.5*cm])
            top_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
                ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE",     (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS",(0, 1),(-1,-1), [colors.HexColor("#fdf9ff"), colors.white]),
                ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#d0c0f0")),
                ("TOPPADDING",   (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ]))
            elements.append(top_table)
            elements.append(Spacer(1, 18))

        # ── Full Student Results ───────────────────────────────────────────
        elements.append(Paragraph("Consolidated Result Sheet", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        result_header = ["#", "Student Name", "USN", "Total", "Result"]
        result_rows = [result_header] + [
            [i+1, r.get("Student Name","?"), r.get("USN","?"), r.get("Total",0), r.get("Result","?")]
            for i, r in enumerate(raw)
        ]

        def row_style(row_idx, result):
            if result == "PASS":
                return colors.HexColor("#d1fae5")
            return colors.HexColor("#fee2e2")

        result_table = Table(result_rows, colWidths=[1*cm, 6*cm, 4*cm, 2.5*cm, 2.5*cm])
        style_cmds = [
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#0369a1")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#c0d0e0")),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]
        for i, r in enumerate(raw):
            bg = colors.HexColor("#d1fae5") if r.get("Result") == "PASS" else colors.HexColor("#fee2e2")
            style_cmds.append(("BACKGROUND", (0, i+1), (-1, i+1), bg))

        result_table.setStyle(TableStyle(style_cmds))
        elements.append(result_table)

        doc.build(elements)
        return FileResponse(
            path=report_path,
            filename=f"result_report_{filename.replace('.pdf','')}.pdf",
            media_type="application/pdf"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)