from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from app.api.v1 import router as api_v1_router

app = FastAPI(title="American Board of AI Exam System", version="0.1.0")


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,
    allow_methods=["*"],              
    allow_headers=["*"],              
)

app.include_router(api_v1_router, prefix="/api/v1")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}