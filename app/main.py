from __future__ import annotations

from fastapi import FastAPI

from .classifier import classify_ticket
from .models import TicketIn, TriageOut

app = FastAPI(
    title="AI Ticket Triage Agent",
    description="Classify support tickets with Groq and draft a reply.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageOut)
def triage(ticket: TicketIn) -> TriageOut:
    return classify_ticket(ticket)
