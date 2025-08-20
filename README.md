# Complete Project Context
my api :
AI Data Agent is a modular, LLM-powered automation system for data analysis, web scraping, and image processing. It uses large language models (LLMs) to understand user requests, generate Python code, and execute it securely in a sandbox.

---

## How It Works

### Input
- **Mandatory:** `questions.txt` describing the user’s task.
- **Optional:** CSV files, images, or other data files.
- **Automatic Extraction:** URLs, keywords, and entities from the questions.

### Processing Flow
1. **Ingest Request:** Build a manifest of all files and extract relevant URLs/keywords.
2. **LLM-driven Orchestration:** Use an LLM classifier to select the appropriate workflow (data analysis, web scraping, image analysis, or general-purpose).
3. **Dynamic Workflow Selection:** Registry of specialized workflows, with a fallback to a robust DynamicCodeExecutionWorkflow for any request within the syllabus boundaries.
4. **Code Generation & Execution Loop:** Generate Python code using LLMs, execute it in a secure sandbox, and retry/repair if errors occur.
5. **Output Formatting & Validation:** Validate outputs against requested formats and enforce size/security limits.

---

## API Usage

Your AI Data Agent exposes a single API endpoint:

**POST https://app.example.com/api/**

- **Required:** `questions.txt` (task description)
- **Optional:** Any number of files (e.g., CSV, images)

**Example:**
```bash
curl "https://app.example.com/api/" \
  -F "questions.txt=@question.txt" \
  -F "image.png=@image.png" \
  -F "data.csv=@data.csv"
```

**Response:**  
A JSON object containing the analysis results, workflow type, and any generated artifacts.

**Note:**  
- `questions.txt` must always be included.
- Additional files are optional and can be any supported format.

---

## Recent Improvements & Optimizations

### PromptManager (`core/prompt_manager.py`)
- Manages prompt templates for all workflows.
- Implements token counting and optimization (using `tiktoken`).
- Caches prompts and responses for efficiency.
- Validates output formats before accepting LLM responses.

### ErrorHandler (`core/error_handler.py`)
- Categorizes and tracks errors (syntax, runtime, import, memory, timeout, validation, permission, unknown).
- Implements robust retry logic (up to 3 attempts).
- Generates error summaries for reporting and debugging.
- Suggests fixes for common errors.

### Enhanced CodeGenerator (`core/code_generator.py`)
- Uses optimized, context-aware prompts from PromptManager.
- Summarizes context to reduce token usage and cost.
- Implements robust error handling via ErrorHandler.
- Validates outputs before returning results.
- Caches code generation results for repeated tasks.

---

## Technical Highlights

- **LLM Orchestration:** Uses LangChain and OpenAI (or open-source LLMs like Phi-3) for code generation, workflow selection, and code repair.
- **Sandbox Execution:** Runs generated code in a secure, isolated environment with strict resource limits (512MB RAM, 3-minute timeout).
- **Supported Libraries:** pandas, numpy, matplotlib, seaborn, requests, beautifulsoup4, bs4, duckdb, scipy, Pillow, opencv-python, etc.
- **Retry & Repair Logic:** Up to 3 attempts per request, with LLM-powered code repair using error traces.
- **Security:** Import validation, process isolation, restricted file system access, and output size limits (100KB).

---

## Workflow Architecture

Each workflow implements:
- `plan()` – Analyze request and create an execution plan.
- `generate_code()` – Generate Python code using LLM prompts.
- `execute()` – Run code in the sandbox.
- `validate()` – Check output format and constraints.
- `repair()` – Fix code using error feedback.

### Specialized Workflows
- **DataAnalysisWorkflow:** For CSV/statistical analysis, including column type inference and data cleaning.
- **WebScrapingWorkflow:** For URL processing, HTML table detection, and navigation (pagination, multi-step).
- **ImageAnalysisWorkflow:** For image processing, OCR, and basic computer vision tasks.
- **DynamicCodeExecutionWorkflow:** General-purpose fallback for any request within the first 6 weeks’ syllabus.

---

## Syllabus Boundaries (Project 2)
Your agent only handles requests from the first 6 weeks:
- **Dev tools:** VS Code, Git, Bash, JSON, Excel, DBs
- **Deployment tools:** Markdown, APIs, Docker, hosting, FastAPI, CI/CD
- **LLMs:** Prompting, embeddings, RAG, function calling, sentiment/extraction
- **Data sourcing:** API scraping, web scraping, PDFs, scheduling
- **Data prep:** Cleaning, transformation, OpenRefine, profiling, parsing JSON, media handling
- **Data analysis:** Excel/Python/SQL analysis, geospatial, network analysis

Requests outside these topics (e.g., advanced visualization, deep learning) are politely rejected.

---

## File Structure
```
core/
├── orchestrator.py          # Main request handler
├── classifier.py            # LLM workflow selection
├── code_generator.py        # Enhanced code generation with token optimization
├── error_handler.py         # Error categorization, retry logic, reporting
├── prompt_manager.py        # Prompt templates, token counting, caching
└── sandbox_executor.py      # Secure execution environment

workflows/
├── base.py                  # Abstract workflow interface
├── dynamic_code_execution.py# General-purpose workflow
├── data_analysis.py         # CSV analysis workflow
├── web_scraping.py          # Web scraping workflow
└── image_analysis.py        # Image analysis workflow
```

---

## Security & Limits
- Sandbox execution with 512MB memory limit
- 3-minute total execution timeout
- Process isolation and restricted file system access
- Import validation against whitelist
- Output size limit (100KB)

---

## Python Package Requirements
```
pandas
numpy
matplotlib
seaborn
requests
beautifulsoup4
bs4
duckdb
scipy
Pillow
opencv-python
tiktoken
```

---

## Deployment Strategy
- **Digital Ocean:** Containerized app, resource-optimized droplet, storage, and monitoring.
- **Production Readiness:** Load testing, logging, and monitoring for reliability.

---

## Current State Summary
- Core engine (DynamicCodeExecutionWorkflow + SandboxExecutor) is fully implemented.
- Specialized workflows for CSV, web, and image analysis are in place.
- **PromptManager, ErrorHandler, and Enhanced CodeGenerator** now provide token optimization, robust error handling, output validation, and caching.
- Integration, further prompt refinement, and production deployment are next priorities.

---

## Changelog — Phase 4 (tests, fixes, integration, docs)

Summary of changes and updates made during Phase 4. Use this as a quick reference for what was fixed, why, and where.

- tests: Fixed async test handling and test expectations
  - Files: `tests/test_integration.py`, `tests/test_request_processor.py`, `tests/test_phase2.py`
  - What changed: Added `pytest.mark.asyncio` to async tests, updated mocks to use `AsyncMock`, fixed return vs assert usage, and aligned mock orchestrator responses with the orchestrator output shape.
  - Status: Done

- workflow constructors & inheritance: Ensure BaseWorkflow required args and abstract methods implemented
  - Files: `workflows/data_analysis.py`, `workflows/dynamic_code_execution.py`, `workflows/web_scraping.py`, `workflows/image_analysis.py`
  - What changed: Fixed duplicate/misplaced `__init__` implementations, added `manifest` parameter where required, implemented `get_workflow_type()` on dynamic workflow to satisfy the abstract base class, and corrected `super()` calls.
  - Status: Done

- orchestrator: accept pre-initialized workflows and include request id in responses
  - Files: `core/orchestrator.py`
  - What changed: `LLMOrchestrator.__init__` now accepts an optional `workflows` dict (used by tests to pass prebuilt workflows). `_execute_workflow` uses provided workflow instances first, and `process_request` generates and returns a `request_id` in the response.
  - Status: Done

- tests vs orchestrator contract alignment
  - Files: `tests/test_request_processor.py`, `tests/test_integration.py`
  - What changed: Updated test mocks to match the orchestrator response shape (including `request_id`, `workflow_used`, `result`), and adjusted assertions accordingly.
  - Status: Done

- code/style/indentation fixes
  - Files: `workflows/dynamic_code_execution.py`, `workflows/data_analysis.py`
  - What changed: Fixed indentation issues, removed misplaced constructor snippets that had leaked into methods, and cleaned up prompt-building logic.
  - Status: Done

- README: expanded documentation and API usage
  - Files: `README.md`
  - What changed: Added full API usage example, technical highlights, workflow architecture, security/limits, package requirements, deployment suggestions, and this Phase 4 changelog.
  - Status: Done

Notes and follow-ups:
- The pydantic deprecation warning is raised by a third-party dependency (pydantic) and can be addressed by upgrading code to `ConfigDict` in a later pass or waiting for dependencies to migrate.
- Recommended next steps: run `pytest -q` locally, containerize the app for deployment, and add a lightweight CI job to run tests on each push.
