# AI Ticket Triage Agent
FastAPI + Groq service that classifies support tickets (`billing`, `technical_bug`, `feature_request`, `general_query`), drafts a reply, and sets `escalate_to_human` when confidence is below 0.75.
`POST /triage` accepts `{ticket_id, customer_email, subject, body}`.
Copy `.env.example` to `.env`, set `GROQ_API_KEY`, then run `uvicorn app.main:app --reload` or `python test_tickets.py`.
