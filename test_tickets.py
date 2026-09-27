from __future__ import annotations

import json

from app.classifier import classify_ticket
from app.models import TicketIn

SAMPLES = [
    TicketIn(
        ticket_id="TCK-1001",
        customer_email="billing@example.com",
        subject="Charged twice for September invoice",
        body=(
            "I was billed $149 twice on the same invoice this month. "
            "Please refund the duplicate charge and confirm the next billing date."
        ),
    ),
    TicketIn(
        ticket_id="TCK-1002",
        customer_email="ops@example.com",
        subject="Webhook retries failing with 500 after deploy",
        body=(
            "After yesterday's release, outbound Slack webhooks return HTTP 500. "
            "Retries pile up in the queue and low-confidence tickets never escalate."
        ),
    ),
    TicketIn(
        ticket_id="TCK-1003",
        customer_email="product@example.com",
        subject="Can we add a Zendesk tag to the Slack alert?",
        body=(
            "When a ticket is escalated, we want the Zendesk tag and priority "
            "included in the Slack payload so the on-call can route faster."
        ),
    ),
]


def main() -> None:
    for ticket in SAMPLES:
        result = classify_ticket(ticket)
        print("=" * 72)
        print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
