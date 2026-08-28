from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models import (
    AIFeedback,
    AIPrediction,
    Category,
    CategoryKeyword,
    ClassificationExample,
    Department,
    Employee,
    Ticket,
    TicketHistory,
)

OPEN_STATUSES = ("OPEN", "TRIAGED", "IN_PROGRESS")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_ticket(session: Session, requester_name: str, requester_email: str | None, title: str, description: str) -> Ticket:
    ticket = Ticket(
        requester_name=requester_name.strip(),
        requester_email=requester_email.strip() if requester_email else None,
        title=title.strip(),
        description=description.strip(),
        status="OPEN",
        priority="MEDIUM",
        created_at=utc_now(),
    )
    session.add(ticket)
    session.flush()
    add_history(session, ticket.id, "CREATED", "Chamado criado.")
    return ticket


def add_history(session: Session, ticket_id: int, event_type: str, description: str) -> None:
    session.add(
        TicketHistory(
            ticket_id=ticket_id,
            event_type=event_type,
            description=description,
            created_at=utc_now(),
        )
    )


def get_classifier_data(session: Session) -> tuple[list[dict], dict[str, list[tuple[str, float]]]]:
    rows = session.execute(
        select(
            ClassificationExample.id,
            ClassificationExample.text,
            Category.id.label("category_id"),
            Category.name.label("category_name"),
        )
        .join(Category, Category.id == ClassificationExample.category_id)
        .where(ClassificationExample.active == 1, Category.active == 1)
        .order_by(ClassificationExample.id)
    ).mappings().all()

    keywords = session.execute(
        select(Category.name, CategoryKeyword.keyword, CategoryKeyword.weight)
        .join(Category, Category.id == CategoryKeyword.category_id)
        .where(CategoryKeyword.active == 1, Category.active == 1)
        .order_by(Category.name, CategoryKeyword.id)
    ).all()

    keyword_map: dict[str, list[tuple[str, float]]] = {}
    for category_name, keyword, weight in keywords:
        keyword_map.setdefault(category_name, []).append((keyword, float(weight)))

    return [dict(row) for row in rows], keyword_map


def get_category(session: Session, category_id: int) -> Category:
    category = session.get(Category, category_id)
    if category is None or category.active != 1:
        raise ValueError("Categoria não encontrada.")
    return category


def choose_employee(session: Session, department_id: int) -> tuple[Employee, int]:
    open_count = func.sum(
        case(
            (Ticket.status.in_(OPEN_STATUSES), 1),
            else_=0,
        )
    )

    row = session.execute(
        select(Employee, open_count.label("open_count"))
        .outerjoin(Ticket, Ticket.assigned_employee_id == Employee.id)
        .where(Employee.department_id == department_id, Employee.active == 1)
        .group_by(Employee.id)
        .order_by(open_count.asc(), Employee.id.asc())
        .limit(1)
    ).first()

    if row is None:
        raise RuntimeError("Nenhum analista ativo disponível para o departamento.")

    return row[0], int(row[1] or 0)


def assign_ticket(session: Session, ticket: Ticket, department_id: int, employee_id: int, priority: str) -> None:
    ticket.assigned_department_id = department_id
    ticket.assigned_employee_id = employee_id
    ticket.priority = priority
    ticket.status = "TRIAGED"
    session.flush()


def save_prediction(
    session: Session,
    ticket_id: int,
    category_id: int,
    semantic_score: float,
    nlp_score: float,
    hybrid_score: float,
    margin: float,
    decision_source: str,
    qwen_used: bool,
    needs_review: bool,
    problem_summary: str,
    extracted_location: str | None,
    entities: dict,
    embedding_model: str,
    llm_model: str | None,
) -> AIPrediction:
    prediction = AIPrediction(
        ticket_id=ticket_id,
        predicted_category_id=category_id,
        semantic_score=semantic_score,
        nlp_score=nlp_score,
        hybrid_score=hybrid_score,
        margin=margin,
        decision_source=decision_source,
        qwen_used=int(qwen_used),
        needs_review=int(needs_review),
        problem_summary=problem_summary,
        extracted_location=extracted_location,
        extracted_entities_json=json.dumps(entities, ensure_ascii=False),
        embedding_model=embedding_model,
        llm_model=llm_model,
        created_at=utc_now(),
    )
    session.add(prediction)
    session.flush()
    return prediction


def list_categories(session: Session) -> list[dict]:
    rows = session.execute(
        select(Category.id, Category.name)
        .where(Category.active == 1)
        .order_by(Category.name)
    ).mappings().all()
    return [dict(row) for row in rows]


def get_review_queue(session: Session, limit: int = 1) -> dict:
    reviewed_exists = select(AIFeedback.id).where(
        AIFeedback.prediction_id == AIPrediction.id
    ).exists()

    pending_count = session.scalar(
        select(func.count())
        .select_from(AIPrediction)
        .where(~reviewed_exists)
    ) or 0

    reviewed_count = session.scalar(
        select(func.count(func.distinct(AIFeedback.prediction_id)))
    ) or 0

    rows = session.execute(
        select(
            AIPrediction.id.label("prediction_id"),
            AIPrediction.ticket_id,
            Ticket.requester_name,
            Ticket.title,
            Ticket.description,
            Ticket.created_at,
            Category.id.label("predicted_category_id"),
            Category.name.label("predicted_category"),
            AIPrediction.semantic_score,
            AIPrediction.nlp_score,
            AIPrediction.hybrid_score,
            AIPrediction.margin,
            AIPrediction.decision_source,
            AIPrediction.qwen_used,
            AIPrediction.needs_review,
            AIPrediction.problem_summary,
            AIPrediction.extracted_location,
            AIPrediction.extracted_entities_json,
        )
        .join(Ticket, Ticket.id == AIPrediction.ticket_id)
        .join(Category, Category.id == AIPrediction.predicted_category_id)
        .where(~reviewed_exists)
        .order_by(AIPrediction.created_at.asc(), AIPrediction.id.asc())
        .limit(max(1, min(limit, 200)))
    ).mappings().all()

    items = []
    for row in rows:
        item = dict(row)
        try:
            item["nlp"] = json.loads(item.pop("extracted_entities_json") or "{}")
        except json.JSONDecodeError:
            item["nlp"] = {}
        item["qwen_used"] = bool(item["qwen_used"])
        item["needs_review"] = bool(item["needs_review"])
        items.append(item)

    return {
        "items": items,
        "pending_count": int(pending_count),
        "reviewed_count": int(reviewed_count),
    }


def list_open_tickets(session: Session) -> list[dict]:
    rows = session.execute(
        select(
            Ticket.id,
            Ticket.requester_name,
            Ticket.title,
            Ticket.description,
            Ticket.status,
            Ticket.priority,
            Ticket.created_at,
            Department.name.label("department_name"),
            Employee.name.label("employee_name"),
            Category.name.label("category_name"),
            AIPrediction.problem_summary,
            AIPrediction.extracted_location,
            AIPrediction.qwen_used,
        )
        .outerjoin(Department, Department.id == Ticket.assigned_department_id)
        .outerjoin(Employee, Employee.id == Ticket.assigned_employee_id)
        .outerjoin(AIPrediction, AIPrediction.ticket_id == Ticket.id)
        .outerjoin(Category, Category.id == AIPrediction.predicted_category_id)
        .where(Ticket.status.in_(OPEN_STATUSES))
        .order_by(Ticket.created_at.asc(), Ticket.id.asc())
    ).mappings().all()
    return [dict(row) for row in rows]


def resolve_ticket(session: Session, ticket_id: int) -> Ticket:
    ticket = session.get(Ticket, ticket_id)
    if ticket is None:
        raise ValueError("Chamado não encontrado.")
    if ticket.status == "RESOLVED":
        return ticket

    ticket.status = "RESOLVED"
    ticket.resolved_at = utc_now()
    add_history(session, ticket.id, "RESOLVED", "Chamado marcado como concluído.")
    session.flush()
    return ticket


def get_dashboard(session: Session) -> dict:
    open_count = session.scalar(
        select(func.count()).select_from(Ticket).where(Ticket.status.in_(OPEN_STATUSES))
    ) or 0
    resolved_count = session.scalar(
        select(func.count()).select_from(Ticket).where(Ticket.status == "RESOLVED")
    ) or 0
    qwen_count = session.scalar(
        select(func.count()).select_from(AIPrediction).where(AIPrediction.qwen_used == 1)
    ) or 0
    prediction_count = session.scalar(select(func.count()).select_from(AIPrediction)) or 0
    reviewed_exists = select(AIFeedback.id).where(
        AIFeedback.prediction_id == AIPrediction.id
    ).exists()
    review_count = session.scalar(
        select(func.count()).select_from(AIPrediction).where(~reviewed_exists)
    ) or 0

    return {
        "open_tickets": int(open_count),
        "resolved_tickets": int(resolved_count),
        "predictions": int(prediction_count),
        "qwen_used": int(qwen_count),
        "qwen_rate": round((qwen_count / prediction_count * 100) if prediction_count else 0.0, 1),
        "needs_review": int(review_count),
    }


def get_team_load(session: Session) -> list[dict]:
    open_count = func.sum(
        case(
            (Ticket.status.in_(OPEN_STATUSES), 1),
            else_=0,
        )
    )

    rows = session.execute(
        select(
            Employee.id,
            Employee.name,
            Department.name.label("department_name"),
            open_count.label("open_tickets"),
        )
        .join(Department, Department.id == Employee.department_id)
        .outerjoin(Ticket, Ticket.assigned_employee_id == Employee.id)
        .where(Employee.active == 1)
        .group_by(Employee.id, Department.name)
        .order_by(Department.name, open_count.asc(), Employee.name)
    ).mappings().all()

    return [
        {
            **dict(row),
            "open_tickets": int(row["open_tickets"] or 0),
        }
        for row in rows
    ]


def save_feedback(session: Session, prediction_id: int, is_correct: bool, corrected_category_id: int | None, notes: str | None) -> AIFeedback:
    prediction = session.get(AIPrediction, prediction_id)
    if prediction is None:
        raise ValueError("Previsão não encontrada.")

    if is_correct:
        corrected_category_id = None
    else:
        if corrected_category_id is None:
            raise ValueError("Informe a categoria correta para uma classificação incorreta.")
        category = session.get(Category, corrected_category_id)
        if category is None or category.active != 1:
            raise ValueError("Categoria corrigida não encontrada.")

    feedback = session.scalar(
        select(AIFeedback)
        .where(AIFeedback.prediction_id == prediction_id)
        .order_by(AIFeedback.id.desc())
        .limit(1)
    )

    if feedback is None:
        feedback = AIFeedback(
            prediction_id=prediction_id,
            is_correct=int(is_correct),
            corrected_category_id=corrected_category_id,
            notes=notes.strip() if notes else None,
            created_at=utc_now(),
        )
        session.add(feedback)
    else:
        feedback.is_correct = int(is_correct)
        feedback.corrected_category_id = corrected_category_id
        feedback.notes = notes.strip() if notes else None
        feedback.created_at = utc_now()

    prediction.needs_review = 0
    session.flush()
    return feedback
