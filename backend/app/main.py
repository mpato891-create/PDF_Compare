# backend/app/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
from datetime import datetime
import logging
from io import BytesIO

# ✅ مكتبة الترجمة الجديدة
from deep_translator import GoogleTranslator

# ✅ Import our modules
from .utils import extract_text, get_document_metadata
from .ai_comparator import compare_with_ai

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= Response Models =============

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

class TranslationRequest(BaseModel):
    text: str

# ============= Application Initialization =============
app = FastAPI(
    title="🔍 Smart AI Contract Comparator",
    description="AI-Powered Contract Comparison Engine",
    version="3.1.0"
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
    """Extracts a simple summary based on emojis."""
    summary = {
        "missing_count": 0,
        "modified_count": 0,
        "additional_count": 0,
    }
    try:
        if report:
            summary["missing_count"] = report.count("❌")
            summary["modified_count"] = report.count("🔄")
            summary["additional_count"] = report.count("➕")
    except Exception as e:
        logger.warning(f"Failed to extract summary: {e}")
    return summary

def translate_large_text(text: str) -> str:
    """
    Translates large text by splitting into chunks suitable for Google Translate.
    This prevents timeouts or length errors.
    """
    if not text: 
        return ""
        
    translator = GoogleTranslator(source='auto', target='ar')
    chunks = []
    # نقسم النص كل 4500 حرف تقريباً (حدود جوجل الآمنة)
    max_chunk = 4500
    
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i + max_chunk]
        try:
            translated_chunk = translator.translate(chunk)
            chunks.append(translated_chunk)
        except Exception as e:
            logger.error(f"Translation failed for chunk: {e}")
            chunks.append(chunk) # نبقي النص الأصلي في حال الفشل
            
    return "\n".join(chunks)

async def _handle_extraction(file: UploadFile):
    """Shared logic for extraction endpoints"""
    start_time = datetime.now()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        file_content, filename = await _process_file_content(file)
        mock_file_stream = BytesIO(file_content)
        mock_file_stream.filename = filename

        # Extract text using our robust utils
        text = await extract_text(mock_file_stream, filename)

        # Extract metadata
        mock_file_stream.seek(0)
        metadata_dict = await asyncio.to_thread(get_document_metadata, mock_file_stream, filename)

        extraction_time = (datetime.now() - start_time).total_seconds() * 1000
        metadata_dict['extraction_time_ms'] = extraction_time

        return ExtractionResponse(
            text=text,
            metadata=DocumentMetadataResponse(**metadata_dict),
            preview_length=len(text[:5000]),
            total_length=len(text),
            extraction_time_ms=extraction_time
        )

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

# ============= Endpoints =============

@app.get("/", tags=["health"], response_model=HealthResponse)
async def root():
    uptime = (datetime.now() - START_TIME).total_seconds()
    return HealthResponse(
        status="healthy",
        version="3.1.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime
    )

# ✅ Endpoint for Translation Button
@app.post("/translate-report", tags=["translation"])
async def translate_report_endpoint(request: TranslationRequest):
    """Receives English text and returns Arabic translation (chunked safely)"""
    try:
        logger.info("🌍 Translation request received...")
        # نستخدم thread منفصل لأن الترجمة قد تأخذ وقتاً
        translated_text = await asyncio.to_thread(translate_large_text, request.text)
        return {"translated_text": translated_text}
    except Exception as e:
        logger.error(f"Translation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract", tags=["extraction"], response_model=ExtractionResponse)
async def extract_full_text(file: UploadFile = File(...)):
    return await _handle_extraction(file)

# ✅ Compatibility Alias for Frontend
@app.post("/extract-preview", tags=["extraction"], response_model=ExtractionResponse)
async def extract_preview_legacy(file: UploadFile = File(...)):
    return await _handle_extraction(file)

@app.post("/compare", tags=["comparison"], response_model=ComparisonResponse)
async def compare_contracts(
    standard: UploadFile = File(..., description="Master/Standard contract"),
    other: UploadFile = File(..., description="Contract to compare")
):
    start_time = datetime.now()
    
    if not standard.filename or not other.filename:
        raise HTTPException(status_code=400, detail="Both files are required")

    try:
        logger.info(f"Starting AI comparison: {standard.filename} vs {other.filename}")

        # 1. Read Files
        standard_content, standard_filename = await _process_file_content(standard)
        other_content, other_filename = await _process_file_content(other)
        
        std_stream = BytesIO(standard_content)
        oth_stream = BytesIO(other_content)

        # 2. Extract Text (Concurrent)
        standard_text, other_text = await asyncio.gather(
            extract_text(std_stream, standard_filename),
            extract_text(oth_stream, other_filename)
        )

        # Basic Validation
        if len(standard_text) < 50 or len(other_text) < 50:
            raise HTTPException(status_code=422, detail="One of the documents is empty or unreadable.")

        # 3. AI Comparison
        report = await asyncio.to_thread(compare_with_ai, standard_text, other_text)

        # 4. Generate Summary stats
        summary = _extract_summary_from_report(report)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return ComparisonResponse(
            report=report,
            summary=summary,
            standard_info={"filename": standard_filename, "length": len(standard_text)},
            other_info={"filename": other_filename, "length": len(other_text)},
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison process failed: {str(e)}")
