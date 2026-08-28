from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import ROOT_DIR
from app.database import get_session
from app.repositories import get_dashboard, get_review_queue, get_team_load, list_categories, list_open_tickets, resolve_ticket, save_feedback
from app.schemas import FeedbackRequest, TicketCreateRequest
from app.services.ticket_ai import TicketAIService
from app.services.ticket_service import TicketService

app = FastAPI(title="AI HelpDesk SQL", version="1.1.0")
STATIC_DIR = ROOT_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_ticket_service() -> TicketService:
    return TicketService(TicketAIService())


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard(session: Session = Depends(get_session)):
    return get_dashboard(session)


@app.get("/api/tickets/open")
def open_tickets(session: Session = Depends(get_session)):
    return {"tickets": list_open_tickets(session)}


@app.get("/api/categories")
def categories(session: Session = Depends(get_session)):
    return {"categories": list_categories(session)}


@app.get("/api/reviews/pending")
def pending_reviews(limit: int = 1, session: Session = Depends(get_session)):
    return get_review_queue(session, limit=limit)


@app.post("/api/tickets")
def create_ticket(payload: TicketCreateRequest, session: Session = Depends(get_session)):
    try:
        return get_ticket_service().create_and_triage(
            session,
            payload.requester_name,
            payload.requester_email,
            payload.title,
            payload.description,
        )
    except Exception as exc:
        session.rollback()
        raise HTTPException(
            status_code=500,
            detail="Falha interna ao processar o chamado. Verifique o terminal do servidor.",
        ) from exc


@app.patch("/api/tickets/{ticket_id}/resolve")
def complete_ticket(ticket_id: int, session: Session = Depends(get_session)):
    try:
        ticket = resolve_ticket(session, ticket_id)
        session.commit()
        return {"ticket_id": ticket.id, "status": ticket.status, "resolved_at": ticket.resolved_at}
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/team")
def team(session: Session = Depends(get_session)):
    return {"team": get_team_load(session)}


@app.post("/api/predictions/{prediction_id}/feedback")
def feedback(prediction_id: int, payload: FeedbackRequest, session: Session = Depends(get_session)):
    try:
        item = save_feedback(
            session,
            prediction_id,
            payload.is_correct,
            payload.corrected_category_id,
            payload.notes,
        )
        session.commit()
        return {"feedback_id": item.id}
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
