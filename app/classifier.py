from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .models import Category, TicketIn, TriageOut

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
PREFERRED_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODELS = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)
ESCALATE_BELOW = 0.75
CATEGORIES = {"billing", "technical_bug", "feature_request", "general_query"}

SYSTEM = """You are a support-ticket triage engine. Return ONE JSON object only:
{"category": "billing|technical_bug|feature_request|general_query", "confidence": <0.0-1.0>, "draft_response": "<professional reply>"}

Rules:
- Classify from the ticket subject and body only.
- confidence is how sure you are of the category, from 0.0 to 1.0.
- draft_response is a short, professional customer-facing reply. Do not invent account facts, refunds, or ETAs.
- If the ticket is mixed or thin, pick the closest category and lower confidence.
"""

_resolved_model: str | None = None


def classify_ticket(ticket: TicketIn) -> TriageOut:
    data, model = _complete_json(ticket)
    category = _category(data.get("category"))
    confidence = _confidence(data.get("confidence"))
    draft = str(data.get("draft_response") or "").strip() or _fallback_draft(category)
    return TriageOut(
        ticket_id=ticket.ticket_id,
        category=category,
        confidence=confidence,
        draft_response=draft,
        escalate_to_human=confidence < ESCALATE_BELOW,
        model=model,
    )


def _complete_json(ticket: TicketIn) -> tuple[dict[str, Any], str]:
    global _resolved_model
    api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"')
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    user = (
        f"ticket_id: {ticket.ticket_id}\n"
        f"customer_email: {ticket.customer_email}\n"
        f"subject: {ticket.subject}\n"
        f"body: {ticket.body}"
    )
    models = [_resolved_model or PREFERRED_MODEL]
    for candidate in FALLBACK_MODELS:
        if candidate not in models:
            models.append(candidate)

    last_error: Exception | None = None
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            _resolved_model = model
            return _parse_json(raw), model
        except Exception as exc:
            last_error = exc
            if "model_not_found" in str(exc).lower() or "does not exist" in str(exc).lower():
                continue
            raise
    raise last_error or RuntimeError("Groq classification failed")


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(match.group(0)) if match else {}
    return parsed if isinstance(parsed, dict) else {}


def _category(raw: object) -> Category:
    value = str(raw or "general_query").strip().lower().replace(" ", "_")
    if value not in CATEGORIES:
        return "general_query"
    return value  # type: ignore[return-value]


def _confidence(raw: object) -> float:
    try:
        return max(0.0, min(1.0, float(raw)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _fallback_draft(category: Category) -> str:
    return (
        f"Thanks for writing in. We have logged this as a {category.replace('_', ' ')} request "
        "and will follow up shortly with the next step."
    )
