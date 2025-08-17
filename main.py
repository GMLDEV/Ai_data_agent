from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import uvicorn

# Import our modules
from core.file_processor import FileProcessor
from sandbox.executor import SandboxExecutor
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LLM Orchestrator",
    description="AI-powered data analysis and task automation system",
    version="1.0.0"
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
sandbox_executor = SandboxExecutor(
    settings.sandbox_memory_limit,
    settings.sandbox_cpu_limit,
    settings.max_execution_time
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Orchestrator API",
        "version": "1.0.0",
        "status": "running"
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
        result = sandbox_executor.test_sandbox()
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
        
        result = sandbox_executor.execute_simple(code)
        return {
            "message": "Code execution completed",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Code execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this new endpoint to main.py after the existing endpoints

@app.post("/process-request")
async def process_request(
    questions: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    """Main LLM-powered request processing endpoint"""
    try:
        # Process uploaded files
        file_dict = {}
        if files:
            for file in files:
                content = await file.read()
                file_dict[file.filename] = content
        
        # Process request with LLM orchestrator
        result = orchestrator.process_request(questions, file_dict)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Also add this import at the top of main.py
from core.orchestrator import LLMOrchestrator

# And replace the old component initialization with:
orchestrator = LLMOrchestrator()
                
if __name__ == "__main__":
    logger.info("Starting LLM Orchestrator server...")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info"
    )