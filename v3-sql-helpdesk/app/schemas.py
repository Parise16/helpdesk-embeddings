from __future__ import annotations

from pydantic import BaseModel, Field


class TicketCreateRequest(BaseModel):
    requester_name: str = Field(min_length=2, max_length=120)
    requester_email: str | None = Field(default=None, max_length=180)
    title: str = Field(min_length=3, max_length=180)
    description: str = Field(min_length=5, max_length=4000)


class FeedbackRequest(BaseModel):
    is_correct: bool
    corrected_category_id: int | None = None
    notes: str | None = Field(default=None, max_length=1000)
