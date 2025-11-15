# demo_langgraph_langsmith_moderated_qa_traced.py
"""
Moderated Q&A (LangGraph + LangSmith)
- Classify -> (block | sanitize) -> answer -> review -> (finalize | rewrite)
- PII mask + safety classification + review/revise loop
- Tracing: set LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY in your .env
- Checkpointing: sqlite file keeps state for resume/inspection
"""

from __future__ import annotations
import os, json, re
from typing import TypedDict, Literal

from dotenv import load_dotenv
load_dotenv()

# ---- Tracing / Project naming (LangSmith) ----
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
# Either LANGSMITH_PROJECT or LANGCHAIN_PROJECT is respected by LangSmith.
os.environ.setdefault("LANGCHAIN_PROJECT", "Session10-LG+LS-ModeratedQA")

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable

# ---- Models (env overridable) ----
LLM_CLASS = ChatOpenAI(model=os.getenv("OPENAI_MODEL_CLASS", "gpt-4o-mini"), temperature=0)
LLM_ANSWER = ChatOpenAI(model=os.getenv("OPENAI_MODEL_ANS", "gpt-4o"), temperature=0)
LLM_REVIEW = ChatOpenAI(model=os.getenv("OPENAI_MODEL_REV", "gpt-4o-mini"), temperature=0)

# ---- Simple PII regex (emails + phone-like strings) ----
PII = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})|(\+?\d[\d\- ]{7,}\d)")

# ---- State ----
class QAState(TypedDict, total=False):
    query: str            # original user input
    safe_query: str       # redacted version
    decision: Literal["block", "ok", "revise"]
    draft: str            # current answer draft
    feedback: str         # review feedback (if any)

# ---- Helpers ----
def _json_or_default(text: str, default: dict) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return default

# ---- Nodes ----
def classify(state: QAState) -> QAState:
    """Fast guard: classify if unsafe; block early if needed."""
    q = state["query"]
    prompt = (
        'Classify if unsafe (e.g., PII harvesting, explicit secrets) or safe.\n'
        'Return EXACT JSON: {"decision":"block"} or {"decision":"ok"}\n'
        f"User input:\n{q}"
    )
    res = LLM_CLASS.invoke(prompt)
    decision = _json_or_default(res.content, {"decision": "ok"}).get("decision", "ok")

    # If blocked, set a safe message now so finalize can display something helpful
    if decision == "block":
        return {
            "decision": "block",
            "draft": "Your request appears unsafe (policy/PII). I can’t comply.",
        }
    return {"decision": "ok"}

@traceable(name="sanitize")  # traced in LangSmith
def sanitize(state: QAState) -> QAState:
    """Mask simple PII patterns before answering."""
    q = state["query"]
    safe = PII.sub("[REDACTED]", q)
    return {"safe_query": safe}

def answer(state: QAState) -> QAState:
    """Answer concisely using sanitized query when present."""
    q = state.get("safe_query") or state["query"]
    res = LLM_ANSWER.invoke(f"Answer concisely (<=6 sentences). Avoid echoing PII:\n{q}")
    return {"draft": res.content}

def review(state: QAState) -> QAState:
    """Gate on clarity + PII echoes; may request a rewrite."""
    d = state.get("draft", "")
    prompt = (
        'Review for clarity (<=6 sentences) and ensure no PII echo.\n'
        'If acceptable, respond with JSON: {"decision":"ok","feedback":"..."}.\n'
        'If not, respond with JSON: {"decision":"revise","feedback":"what to fix ..."}.\n'
        f"Draft:\n{d}"
    )
    res = LLM_REVIEW.invoke(prompt)
    parsed = _json_or_default(res.content, {"decision": "ok", "feedback": "Looks good."})
    return {
        "decision": parsed.get("decision", "ok"),
        "feedback": parsed.get("feedback", "Looks good."),
    }

def rewrite(state: QAState) -> QAState:
    """Revise draft per feedback constraints."""
    d = state.get("draft", "")
    fb = state.get("feedback", "")
    res = LLM_ANSWER.invoke(
        "Revise the draft per feedback. Keep <=6 sentences. Do not add/echo PII.\n"
        f"Feedback:\n{fb}\n\nDraft:\n{d}\n"
    )
    return {"draft": res.content}

def finalize(state: QAState) -> QAState:
    """No-op node; state is already populated. Useful anchor for tracing/checkpoints."""
    return {}

# ---- Routers ----
def route_after_classify(state: QAState) -> Literal["finalize", "sanitize"]:
    return "finalize" if state.get("decision") == "block" else "sanitize"

def route_after_review(state: QAState) -> Literal["finalize", "rewrite"]:
    return "finalize" if state.get("decision") == "ok" else "rewrite"

# ---- Graph ----
graph = StateGraph(QAState)
graph.add_node("classify", classify)
graph.add_node("sanitize", sanitize)
graph.add_node("answer", answer)
graph.add_node("review", review)
graph.add_node("rewrite", rewrite)
graph.add_node("finalize", finalize)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_after_classify, {
    "finalize": "finalize",
    "sanitize": "sanitize",
})
graph.add_edge("sanitize", "answer")
graph.add_edge("answer", "review")
graph.add_conditional_edges("review", route_after_review, {
    "finalize": "finalize",
    "rewrite": "rewrite",
})
graph.add_edge("rewrite", "review")
graph.add_edge("finalize", END)

# ---- Run with persistent checkpoint (so you can inspect/resume) ----
if __name__ == "__main__":
    start: QAState = {
        "query": "My email is john.smith@example.com. Draft a short intro to share with a client.",
    }
    cfg = {"configurable": {"thread_id": "lg-ls-moderatedqa-001"}}

    # Keep compile+stream INSIDE the context so the DB stays open during the run.
    with SqliteSaver.from_conn_string("lg_ls_moderatedqa.sqlite") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        print("=== STREAM ===")
        for ev in app.stream(start, config=cfg):
            print(ev)

        final = app.get_state(cfg).values
        print("\n--- FINAL ---")
        print("Decision:", final.get("decision"))
        print("Answer:\n", final.get("draft", "<no draft>"))
