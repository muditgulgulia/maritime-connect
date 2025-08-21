import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.auth.routes import router as auth_router
from app.routes.protected import router as protected_router
from app.routes.qa import router as qa_router
from app.routes.ingest import router as ingest

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan context manager replaces on_event handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up…")
    # e.g., initialize DB connections, load models
    yield
    logger.info("Application shutting down…")
    # cleanup here

app = FastAPI(
    title="Maritime Connect API",
    description="API for machinery maintenance and diagnostics",
    version="1.0.0",
    lifespan=lifespan  # register lifespan
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Mount routers
app.include_router(auth_router)
app.include_router(protected_router)
app.include_router(qa_router)
app.include_router(ingest)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
