from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field

Category = Literal["billing", "technical_bug", "feature_request", "general_query"]


class TicketIn(BaseModel):
    ticket_id: str
    customer_email: EmailStr
    subject: str
    body: str


class TriageOut(BaseModel):
    ticket_id: str
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    draft_response: str
    escalate_to_human: bool
    model: str
