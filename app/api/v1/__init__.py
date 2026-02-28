from fastapi import APIRouter
from app.api.v1.endpoints import auth, exams
router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(exams.router, tags=["exams"])
