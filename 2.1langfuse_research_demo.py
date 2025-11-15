# demo2_langgraph_langfuse_moderated_qa.py
from __future__ import annotations
import os, json, re
from typing import TypedDict, Literal

from dotenv import load_dotenv
load_dotenv()

# --- Langfuse (v3) callback for LangChain ---
# pip install -U langfuse>=3 langchain-openai langgraph python-dotenv

# --- LangChain / LangGraph ---
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------- LLMs ----------
LLM_CLASS = ChatOpenAI(model=os.getenv("OPENAI_MODEL_CLASS", "gpt-4o-mini"), temperature=0)
LLM_ANSWER = ChatOpenAI(model=os.getenv("OPENAI_MODEL_ANS", "gpt-4o"), temperature=0)
LLM_REVIEW = ChatOpenAI(model=os.getenv("OPENAI_MODEL_REV", "gpt-4o-mini"), temperature=0)

# Langfuse callback handler (reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST)

from langfuse.langchain import CallbackHandler

lf_cb = CallbackHandler(  # optional kwargs: user_id, session_id, tags, metadata
)

# lf_cb = LangfuseCallbackHandler(
#     # optional: set a project or tags to group runs in Langfuse UI
#     tags=["session10", "moderated-qa", "langgraph"]
# )

# ---------- Simple PII sanitizer ----------
PII = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})|(\+?\d[\d\- ]{7,}\d)")

# ---------- Shared state ----------
class QAState(TypedDict, total=False):
    query: str
    safe_query: str
    decision: Literal["block", "ok", "revise"]
    draft: str
    feedback: str
    error: str

# ---------- Nodes ----------
def classify(state: QAState) -> QAState:
    q = state["query"]
    prompt = (
        'Classify the user input as safe or unsafe (PII harvesting, explicit secrets, criminal).\n'
        'Return EXACT JSON: {"decision":"block"} or {"decision":"ok"}\n'
        f"User input:\n{q}"
    )
    res = LLM_CLASS.invoke(prompt, config={"callbacks": [lf_cb], "tags": ["classify"]})
    try:
        d = json.loads(res.content).get("decision", "ok")
    except Exception:
        d = "ok"
    return {"decision": d}

def sanitize(state: QAState) -> QAState:
    q = state["query"]
    safe = PII.sub("[REDACTED]", q)
    return {"safe_query": safe}

def answer(state: QAState) -> QAState:
    q = state.get("safe_query") or state["query"]
    res = LLM_ANSWER.invoke(f"Answer concisely (≤6 sentences):\n{q}",
                            config={"callbacks": [lf_cb], "tags": ["answer"]})
    return {"draft": res.content}

def review(state: QAState) -> QAState:
    d = state.get("draft", "")
    prompt = (
        'You are a strict editor. Check clarity (≤6 sentences) and ensure no PII is echoed.\n'
        'Return JSON: {"decision":"ok","feedback":"..."} or {"decision":"revise","feedback":"..."}\n'
        f"Draft:\n{d}"
    )
    res = LLM_REVIEW.invoke(prompt, config={"callbacks": [lf_cb], "tags": ["review"]})
    try:
        p = json.loads(res.content)
        decision = p.get("decision", "ok")
        feedback = p.get("feedback", "Looks good.")
    except Exception:
        decision, feedback = "ok", "Looks good."
    return {"decision": decision, "feedback": feedback}

def rewrite(state: QAState) -> QAState:
    d = state.get("draft", "")
    fb = state.get("feedback", "")
    res = LLM_ANSWER.invoke(
        f"Revise the draft per feedback. Do NOT add PII.\nFeedback:\n{fb}\nDraft:\n{d}",
        config={"callbacks": [lf_cb], "tags": ["rewrite"]},
    )
    return {"draft": res.content}

def finalize(_: QAState) -> QAState:
    return {}

# ---------- Routers ----------
def route_after_classify(state: QAState) -> Literal["finalize", "sanitize"]:
    return "finalize" if state.get("decision") == "block" else "sanitize"

def route_after_review(state: QAState) -> Literal["finalize", "rewrite"]:
    return "finalize" if state.get("decision") == "ok" else "rewrite"

# ---------- Graph ----------
graph = StateGraph(QAState)
graph.add_node("classify", classify)
graph.add_node("sanitize", sanitize)
graph.add_node("answer", answer)
graph.add_node("review", review)
graph.add_node("rewrite", rewrite)
graph.add_node("finalize", finalize)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_after_classify, {"finalize": "finalize", "sanitize": "sanitize"})
graph.add_edge("sanitize", "answer")
graph.add_edge("answer", "review")
graph.add_conditional_edges("review", route_after_review, {"finalize": "finalize", "rewrite": "rewrite"})
graph.add_edge("rewrite", "review")
graph.add_edge("finalize", END)

# ---------- Run ----------
if __name__ == "__main__":
    start: QAState = {
        "query": "My email is john.smith@example.com. Draft a short intro email to a client."
    }
    start: QAState = {
        "query": "What is RAG."
    }
    cfg = {"configurable": {"thread_id": "lf-langgraph-moderatedqa-001"}}

    # Persist graph state (lets you resume / inspect); not related to Langfuse
    with SqliteSaver.from_conn_string("lf_moderatedqa.sqlite") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        # Stream steps; Langfuse callback will capture each LLM call automatically
        for ev in app.stream(start, config=cfg):
            print(ev)

        final = app.get_state(cfg).values
        print("\n--- FINAL ---")
        print("Decision:", final.get("decision"))
        print("Answer:\n", final.get("draft", "<no draft>"))
