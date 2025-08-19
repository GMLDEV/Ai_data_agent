from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import uvicorn
import os

# Import our modules
from core.file_processor import FileProcessor
from core.orchestrator import Orchestrator  # Changed from LLMOrchestrator to Orchestrator
from sandbox.executor import SandboxExecutor
from config import settings

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    # Get log level from environment or default to INFO
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Setup root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(name)s | %(levelname)s | [%(filename)s:%(lineno)d] | %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Set specific loggers for detailed debugging
    loggers_config = {
        'core.orchestrator': logging.DEBUG,
        'core.sandbox_executor': logging.DEBUG,
        'core.code_generator': logging.DEBUG,
        'workflows': logging.DEBUG,
        'uvicorn.access': logging.INFO,
        'uvicorn.error': logging.INFO,
    }
    
    for logger_name, level in loggers_config.items():
        logging.getLogger(logger_name).setLevel(level)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

logger.info("🚀 Starting AI Data Agent with comprehensive logging...")

# Create FastAPI app
app = FastAPI(
    title="AI Data Agent",
    description="AI-powered data analysis and task automation system with comprehensive logging",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
file_processor = FileProcessor()
orchestrator = Orchestrator(llm_available=True)  # Use new Orchestrator class
logger.info("✅ Core components initialized successfully")

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "AI Data Agent API with Enhanced Logging",
        "version": "2.0.0",
        "status": "healthy",
        "logging": "comprehensive Docker logging enabled",
        "features": [
            "OpenAI GPT-4 powered code generation",
            "Intelligent retry system",
            "Clean API responses",
            "Comprehensive Docker logging",
            "Task-aware code generation"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"  # Will be dynamic later
    }

@app.post("/test-upload")
async def test_upload(
    files: Optional[List[UploadFile]] = File(None),
    questions: str = Form(default="Test questions")
):
    """Test file upload and processing"""
    try:
        result = {
            "message": "File upload test successful",
            "questions": questions,
            "files_received": []
        }
        
        if files:
            file_dict = {}
            for file in files:
                content = await file.read()
                file_dict[file.filename] = content
                
                result["files_received"].append({
                    "filename": file.filename,
                    "size": len(content),
                    "content_type": file.content_type
                })
            
            # Test file processing
            manifest = file_processor.create_manifest(file_dict, questions)
            result["manifest"] = manifest
        
        return result
        
    except Exception as e:
        logger.error(f"Test upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-sandbox")
async def test_sandbox():
    """Test sandbox execution"""
    try:
        result = orchestrator.sandbox.test_sandbox()
        return {
            "message": "Sandbox test completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Sandbox test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute-code")
async def execute_code(code: str = Form(...)):
    """Execute Python code in sandbox (for testing)"""
    try:
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")
        
        result = orchestrator.sandbox.execute_simple(code)
        return {
            "message": "Code execution completed",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Code execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this new endpoint to main.py after the existing endpoints

# Main API endpoint for evaluation team
@app.post("/api/")
async def api_endpoint(request: Request):
    """
    Main API endpoint for evaluation team.
    Accepts POST requests with multipart form data:
    - questions.txt (required): Contains the questions
    - Additional files (optional): CSV, images, etc.
    """
    import traceback
    from fastapi import Request
    
    try:
        logger.info("📥 Received API request from evaluation team")
        
        # Get the form data
        form = await request.form()
        logger.info(f"📋 Form fields received: {list(form.keys())}")
        
        # Extract questions from questions.txt
        questions_file = form.get("questions.txt")
        if not questions_file:
            logger.error("❌ questions.txt not found in request")
            return JSONResponse(
                status_code=400,
                content={"error": "questions.txt is required"}
            )
        
        # Read questions content
        questions_content = await questions_file.read()
        questions = questions_content.decode('utf-8').strip()
        logger.info(f"📝 Questions received: {questions[:200]}...")
        
        # Process additional files
        file_dict = {}
        for field_name, file_data in form.items():
            if field_name == "questions.txt":
                continue  # Skip questions.txt, already processed
            
            if hasattr(file_data, 'read'):  # It's a file
                content = await file_data.read()
                file_dict[field_name] = content
                logger.info(f"📁 File received: {field_name} ({len(content)} bytes)")
        
        logger.info(f"🗂️ Total files for processing: {len(file_dict)}")
        
        # Process request with orchestrator
        logger.info("🔄 Starting request processing with orchestrator")
        result = orchestrator.process_request(questions, file_dict)
        
        logger.info("✅ Request processing completed successfully")
        return JSONResponse(content=result)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ API request processing failed: {str(e)}\n{tb}")
        
        # Return detailed error for debugging
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": tb if os.getenv('DEBUG') else None
            }
        )

# Main endpoint for evaluators using curl with -F flag
@app.post("/process-request")
async def process_request(request: Request):
    """
    Main endpoint for evaluators using curl with -F flag.
    Accepts POST requests with multipart form data:
    - questions.txt (required): Contains the questions
    - Additional files (optional): CSV, images, etc.
    
    Example: curl "https://app.example.com/process-request" -F "questions.txt=@question.txt" -F "data.csv=@data.csv"
    """
    import traceback
    
    try:
        logger.info("📥 Received process-request from evaluator")
        
        # Get the form data
        form = await request.form()
        logger.info(f"📋 Form fields received: {list(form.keys())}")
        
        # Extract questions from questions.txt field
        questions_file = form.get("questions.txt")
        if not questions_file:
            logger.error("❌ questions.txt not found in form data")
            return JSONResponse(
                status_code=400,
                content={"error": "questions.txt is required in form data"}
            )
        
        # Read questions content
        questions_content = await questions_file.read()
        questions = questions_content.decode('utf-8').strip()
        logger.info(f"📝 Questions received: {questions[:200]}...")
        
        # Process additional files
        file_dict = {}
        for field_name, file_data in form.items():
            if field_name == "questions.txt":
                continue  # Skip questions.txt, already processed
            
            if hasattr(file_data, 'read'):  # It's a file
                content = await file_data.read()
                # Use the original filename if available, otherwise use field name
                filename = getattr(file_data, 'filename', field_name)
                file_dict[filename] = content
                logger.info(f"📁 File received: {filename} ({len(content)} bytes)")
        
        logger.info(f"🗂️ Total files for processing: {len(file_dict)}")
        
        # Process request with orchestrator
        logger.info("🔄 Starting request processing with orchestrator")
        result = orchestrator.process_request(questions, file_dict)
        
        logger.info("✅ Request processing completed successfully")
        return JSONResponse(content=result)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Request processing failed: {str(e)}\n{tb}")
        
        # Try to include workflow and manifest info if possible
        workflow = None
        manifest_type = None
        try:
            workflow = getattr(orchestrator, 'last_workflow', None)
            manifest_type = type(getattr(orchestrator, 'last_manifest', None)).__name__
        except Exception:
            pass
            
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": tb if os.getenv('DEBUG') else None,
                "workflow": str(workflow),
                "manifest_type": str(manifest_type)
            }
        )
                
if __name__ == "__main__":
    logger.info("Starting LLM Orchestrator server...")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info"
    )