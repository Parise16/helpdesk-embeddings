from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import EMBEDDING_MODEL_NAME, OLLAMA_LLM_MODEL
from app.repositories import (
    add_history,
    assign_ticket,
    choose_employee,
    create_ticket,
    get_category,
    save_prediction,
)
from app.services.ticket_ai import TicketAIService


class TicketService:
    def __init__(self, ai_service: TicketAIService) -> None:
        self.ai_service = ai_service

    @staticmethod
    def _message(problem: str, location: str | None, department: str, employee: str, queue_ahead: int) -> str:
        location_part = f" em {location}" if location else ""
        first = f"Seu chamado sobre {problem.lower()}{location_part} foi enviado ao departamento {department}."
        if queue_ahead == 0:
            second = f"{employee} ficará responsável pelo atendimento e você é o próximo chamado da fila desse responsável."
        elif queue_ahead == 1:
            second = f"{employee} ficará responsável pelo atendimento. Atualmente há 1 chamado na sua frente com esse responsável."
        else:
            second = f"{employee} ficará responsável pelo atendimento. Atualmente há {queue_ahead} chamados na sua frente com esse responsável."
        return f"{first} {second}"

    def create_and_triage(
        self,
        session: Session,
        requester_name: str,
        requester_email: str | None,
        title: str,
        description: str,
    ) -> dict:
        ticket = create_ticket(session, requester_name, requester_email, title, description)
        ai_result = self.ai_service.analyze(session, title, description)
        category = get_category(session, ai_result.category_id)
        department = category.department
        employee, queue_ahead = choose_employee(session, department.id)

        assign_ticket(session, ticket, department.id, employee.id, ai_result.priority)

        location = ai_result.analysis.locations[0] if ai_result.analysis.locations else None

        prediction = save_prediction(
            session=session,
            ticket_id=ticket.id,
            category_id=ai_result.category_id,
            semantic_score=ai_result.semantic_score,
            nlp_score=ai_result.nlp_score,
            hybrid_score=ai_result.hybrid_score,
            margin=ai_result.margin,
            decision_source=ai_result.decision_source,
            qwen_used=ai_result.qwen_used,
            needs_review=ai_result.needs_review,
            problem_summary=ai_result.problem_summary,
            extracted_location=location,
            entities=ai_result.analysis.as_dict(),
            embedding_model=EMBEDDING_MODEL_NAME,
            llm_model=OLLAMA_LLM_MODEL if ai_result.qwen_used else None,
        )

        add_history(
            session,
            ticket.id,
            "TRIAGED",
            f"Categoria '{category.name}', departamento '{department.name}', responsável '{employee.name}'.",
        )

        user_message = self._message(
            ai_result.problem_summary,
            location,
            department.name,
            employee.name,
            queue_ahead,
        )

        session.commit()

        return {
            "ticket_id": ticket.id,
            "status": ticket.status,
            "priority": ticket.priority,
            "category": category.name,
            "department": department.name,
            "employee": employee.name,
            "queue_ahead": queue_ahead,
            "problem_summary": ai_result.problem_summary,
            "location": location,
            "qwen_used": ai_result.qwen_used,
            "needs_review": ai_result.needs_review,
            "decision_source": ai_result.decision_source,
            "semantic_score": ai_result.semantic_score,
            "nlp_score": ai_result.nlp_score,
            "hybrid_score": ai_result.hybrid_score,
            "margin": ai_result.margin,
            "prediction_id": prediction.id,
            "user_message": user_message,
            "candidates": [candidate.as_dict() for candidate in ai_result.candidates],
            "nlp": ai_result.analysis.as_dict(),
        }
