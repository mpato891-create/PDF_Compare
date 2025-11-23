# backend/app/main.py
# Main Application - FastAPI

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
from datetime import datetime
import logging
from io import BytesIO

# ✅ Correct relative imports
from .utils import extract_text, get_document_metadata
from .ai_comparator import compare_with_ai

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= Structured Response Models =============

class DocumentMetadataResponse(BaseModel):
    page_count: int
    has_tables: bool
    has_images: bool
    is_scanned: bool
    is_corrupted: bool
    scanned_pages_count: int
    language: str
    structure_type: str
    confidence: float
    extraction_time_ms: float
    corruption_details: str
    ocr_used: bool

class ExtractionResponse(BaseModel):
    text: str
    metadata: DocumentMetadataResponse
    preview_length: int
    total_length: int
    extraction_time_ms: float

class ComparisonResponse(BaseModel):
    report: str
    summary: Dict[str, Any]
    standard_info: Dict[str, Any]
    other_info: Dict[str, Any]
    processing_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    uptime_seconds: float

# ============= Application Initialization =============

# ✅ This is the variable 'app' that uvicorn is looking for!
app = FastAPI(
    title="🔍 Smart Contract Comparator API",
    description="Smart Contract Comparison Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = datetime.now()

# ============= Helper Functions =============

async def _process_file_content(file: UploadFile):
    """Read file content and filename safely"""
    content = await file.read()
    filename = file.filename if file.filename else "unknown_file"
    return content, filename

def _extract_summary_from_report(report: str) -> Dict:
    """Extract summary from the text report"""
    summary = {
        "missing_count": 0,
        "modified_count": 0,
        "additional_count": 0,
    }
    try:
        summary["missing_count"] = report.count("❌")
        summary["modified_count"] = report.count("🔄")
        summary["additional_count"] = report.count("➕")
    except Exception as e:
        logger.warning(f"Failed to extract summary: {e}")
    return summary

# ============= Middleware =============

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - Time: {process_time:.3f}s"
    )
    return response

# ============= Endpoints =============

@app.get("/", tags=["health"], response_model=HealthResponse)
async def root():
    uptime = (datetime.now() - START_TIME).total_seconds()
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime
    )

@app.get("/health", tags=["health"])
async def health_check():
    uptime = (datetime.now() - START_TIME).total_seconds()
    return {
        "status": "healthy",
        "service": "Contract Comparator API",
        "version": "2.0.0",
        "uptime_seconds": uptime,
        "timestamp": datetime.now().isoformat(),
        "features": {
            "text_extraction": "enabled",
            "ai_comparison": "enabled",
            "multi_language": "enabled (Arabic/English)"
        }
    }

@app.post("/extract", tags=["extraction"], response_model=ExtractionResponse)
async def extract_full_text(
    file: UploadFile = File(..., description="PDF or DOCX file to extract")
):
    start_time = datetime.now()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    try:
        # Read content once
        file_content, filename = await _process_file_content(file)
        
        # Create BytesIO stream for processing
        mock_file_stream = BytesIO(file_content)
        mock_file_stream.filename = filename
        
        # Extract text
        text = await extract_text(mock_file_stream, filename)
        
        if text.startswith("[Error:") or text.startswith("[Warning:"):
            raise HTTPException(status_code=422, detail=text)
        
        # Extract metadata
        mock_file_stream.seek(0)
        metadata_dict = await asyncio.to_thread(
            get_document_metadata, 
            mock_file_stream, 
            filename
        )
        
        extraction_time = (datetime.now() - start_time).total_seconds() * 1000
        metadata_dict['extraction_time_ms'] = extraction_time

        return ExtractionResponse(
            text=text,
            metadata=DocumentMetadataResponse(**metadata_dict),
            preview_length=len(text[:5000]),
            total_length=len(text),
            extraction_time_ms=extraction_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

@app.post("/extract-preview", tags=["extraction"])
async def extract_preview(
    file: UploadFile = File(..., description="File to preview"),
    preview_length: int = 20000
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    try:
        file_content, filename = await _process_file_content(file)
        mock_file_stream = BytesIO(file_content)
        mock_file_stream.filename = filename
        
        text = await extract_text(mock_file_stream, filename)

        if text.startswith("[Error:") or text.startswith("[Warning:"):
            return {"text": text, "is_error": True}
        
        return {
            "text": text[:preview_length],
            "total_length": len(text),
            "is_truncated": len(text) > preview_length,
            "filename": filename
        }
    except Exception as e:
        logger.error(f"Preview extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preview extraction failed: {str(e)}")

@app.post("/compare", tags=["comparison"], response_model=ComparisonResponse)
async def compare_contracts(
    standard: UploadFile = File(..., description="Master/Standard contract"),
    other: UploadFile = File(..., description="Contract to compare")
):
    start_time = datetime.now()
    
    if not standard.filename or not other.filename:
        raise HTTPException(status_code=400, detail="Both files are required")
    
    try:
        logger.info(f"Starting comparison: {standard.filename} vs {other.filename}")
        
        # Read contents
        standard_content, standard_filename = await _process_file_content(standard)
        other_content, other_filename = await _process_file_content(other)
        
        standard_stream = BytesIO(standard_content)
        other_stream = BytesIO(other_content)

        # Concurrent Extraction
        standard_text, other_text = await asyncio.gather(
            extract_text(standard_stream, standard_filename),
            extract_text(other_stream, other_filename)
        )
        
        # Validations
        if standard_text.startswith("[Error:") or len(standard_text) < 100:
            raise HTTPException(status_code=422, detail="Standard contract extraction failed or empty")
        if other_text.startswith("[Error:") or len(other_text) < 100:
            raise HTTPException(status_code=422, detail="Comparison contract extraction failed or empty")
        
        # AI Comparison
        report = await asyncio.to_thread(compare_with_ai, standard_text, other_text)
        summary = _extract_summary_from_report(report)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ComparisonResponse(
            report=report,
            summary=summary,
            standard_info={"filename": standard_filename, "length": len(standard_text), "preview": standard_text[:500] + "..."},
            other_info={"filename": other_filename, "length": len(other_text), "preview": other_text[:500] + "..."},
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison process failed: {str(e)}")