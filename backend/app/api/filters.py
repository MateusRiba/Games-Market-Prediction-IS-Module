from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..services.database import get_session
from backend.app.models import Game  # SQLAlchemy model mapeando games_dim

router = APIRouter()

@router.get("/filters/genres")
async def list_genres(db: Session = Depends(get_session)):
    rows = await db.execute(select(Game.genres).distinct())
    return [r[0] for r in rows]