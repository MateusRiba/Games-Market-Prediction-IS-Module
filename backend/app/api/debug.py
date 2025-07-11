# backend/app/api/debug.py
from fastapi import APIRouter, Depends
from sqlalchemy import text
from ..services.database import get_session  # seu AsyncSessionLocal

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/peek")
async def peek_table(limit: int = 5, db = Depends(get_session)):
    """
    Retorna as primeiras N linhas de games_dim em JSON
    (apenas para depuração; não expose em produção final)
    """
    stmt = text(f"SELECT * FROM games_dim LIMIT :lim")
    rows = await db.execute(stmt, {"lim": limit})
    # rows.mappings() converte para listas de dicionários
    return [dict(row) for row in rows.mappings()]
