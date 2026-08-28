from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    active: Mapped[int] = mapped_column(Integer)

    employees: Mapped[list["Employee"]] = relationship(back_populates="department")
    categories: Mapped[list["Category"]] = relationship(back_populates="department")


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id"))
    name: Mapped[str] = mapped_column(String)
    active: Mapped[int] = mapped_column(Integer)

    department: Mapped["Department"] = relationship(back_populates="employees")


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id"))
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    active: Mapped[int] = mapped_column(Integer)

    department: Mapped["Department"] = relationship(back_populates="categories")
    examples: Mapped[list["ClassificationExample"]] = relationship(back_populates="category")
    keywords: Mapped[list["CategoryKeyword"]] = relationship(back_populates="category")


class ClassificationExample(Base):
    __tablename__ = "classification_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String)
    active: Mapped[int] = mapped_column(Integer)

    category: Mapped["Category"] = relationship(back_populates="examples")


class CategoryKeyword(Base):
    __tablename__ = "category_keywords"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    keyword: Mapped[str] = mapped_column(String)
    weight: Mapped[float] = mapped_column(Float)
    active: Mapped[int] = mapped_column(Integer)

    category: Mapped["Category"] = relationship(back_populates="keywords")


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    requester_name: Mapped[str] = mapped_column(String)
    requester_email: Mapped[str | None] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String)
    priority: Mapped[str] = mapped_column(String)
    assigned_department_id: Mapped[int | None] = mapped_column(ForeignKey("departments.id"))
    assigned_employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"))
    created_at: Mapped[str] = mapped_column(String)
    resolved_at: Mapped[str | None] = mapped_column(String)


class AIPrediction(Base):
    __tablename__ = "ai_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), unique=True)
    predicted_category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    semantic_score: Mapped[float] = mapped_column(Float)
    nlp_score: Mapped[float] = mapped_column(Float)
    hybrid_score: Mapped[float] = mapped_column(Float)
    margin: Mapped[float] = mapped_column(Float)
    decision_source: Mapped[str] = mapped_column(String)
    qwen_used: Mapped[int] = mapped_column(Integer)
    needs_review: Mapped[int] = mapped_column(Integer)
    problem_summary: Mapped[str] = mapped_column(Text)
    extracted_location: Mapped[str | None] = mapped_column(String)
    extracted_entities_json: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String)
    llm_model: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(String)


class AIFeedback(Base):
    __tablename__ = "ai_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("ai_predictions.id"))
    is_correct: Mapped[int] = mapped_column(Integer)
    corrected_category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String)


class TicketHistory(Base):
    __tablename__ = "ticket_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"))
    event_type: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String)
