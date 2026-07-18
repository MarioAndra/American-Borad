from fastapi import APIRouter
from app.api.v1.endpoints import auth, exams, adaptive_exams, rag

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(exams.router, tags=["exams"])
router.include_router(adaptive_exams.router, prefix="/phase2", tags=["phase2"])
router.include_router(rag.router, prefix="/rag", tags=["rag"])
