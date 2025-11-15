from __future__ import annotations
import os, json, re, time
from typing import TypedDict, Literal, Callable, Any
from dotenv import load_dotenv

# --- OTel setup ---
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# --- LangChain / LangGraph ---
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

OPENAI_MODEL_CLASS = os.getenv("OPENAI_MODEL_CLASS", "gpt-4o-mini")
OPENAI_MODEL_ANS   = os.getenv("OPENAI_MODEL_ANS",   "gpt-4o")
OPENAI_MODEL_REV   = os.getenv("OPENAI_MODEL_REV",   "gpt-4o-mini")

LLM_CLASS = ChatOpenAI(model=OPENAI_MODEL_CLASS, temperature=0)
LLM_ANSWER= ChatOpenAI(model=OPENAI_MODEL_ANS, temperature=0)
LLM_REVIEW= ChatOpenAI(model=OPENAI_MODEL_REV, temperature=0)

PII = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})|(\+?\d[\d\- ]{7,}\d)")

# ---------- OpenTelemetry init ----------
def init_tracer(service_name="Session10-OTel-ModeratedQA"):
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").rstrip("/")
    if endpoint:
        # e.g. http://localhost:4318  -> exporter will post to /v1/traces
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

tracer = init_tracer()

# ---------- Small tracing helper (decorator for nodes) ----------
def traced(node_name: str):
    def deco(fn: Callable[[Any], Any]):
        def wrapper(state):
            with tracer.start_as_current_span(node_name) as span:
                t0 = time.perf_counter()
                try:
                    # add a few useful attributes
                    span.set_attribute("node.name", node_name)
                    # redact raw input if sensitive; this is a demo:
                    q = state.get("query", "")
                    span.set_attribute("input.length", len(str(q)))

                    out = fn(state)

                    # try to record LLM token usage if present on last call
                    for meta_key in ("response_metadata",):
                        for k in ("token_usage", ):
                            usage = getattr(LLM_ANSWER, meta_key, None)
                            if usage and isinstance(usage, dict):
                                tu = usage.get(k)
                                if tu:
                                    for kk, vv in tu.items():
                                        span.set_attribute(f"llm.{kk}", vv)

                    return out
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, str(e)))
                    raise
                finally:
                    span.set_attribute("latency.ms", round((time.perf_counter() - t0) * 1000, 2))
        return wrapper
    return deco

# ---------- State & nodes ----------
class QAState(TypedDict, total=False):
    query: str
    safe_query: str
    decision: Literal["block","ok","revise"]
    draft: str
    feedback: str

@traced("classify")
def classify(state: QAState) -> QAState:
    q = state["query"]
    prompt = (
        'Classify if unsafe (PII harvesting, explicit secrets) or safe.\n'
        'Return EXACT JSON: {"decision":"block"} or {"decision":"ok"}\n'
        f"User input:\n{q}"
    )
    res = LLM_CLASS.invoke(prompt)
    try:
        d = json.loads(res.content).get("decision","ok")
    except Exception:
        d = "ok"
    return {"decision": d}

@traced("sanitize")
def sanitize(state: QAState) -> QAState:
    q = state["query"]
    safe = PII.sub("[REDACTED]", q)
    return {"safe_query": safe}

@traced("answer")
def answer(state: QAState) -> QAState:
    q = state.get("safe_query") or state["query"]
    res = LLM_ANSWER.invoke(f"Answer concisely:\n{q}")
    # record token usage if available
    usage = getattr(res, "response_metadata", {}).get("token_usage", {})
    span = trace.get_current_span()
    for k, v in usage.items():
        span.set_attribute(f"llm.token_usage.{k}", v)
    return {"draft": res.content}

@traced("review")
def review(state: QAState) -> QAState:
    d = state.get("draft","")
    prompt = (
        'Review for clarity (<=6 sentences) and no PII echo.\n'
        'Return JSON: {"decision":"ok","feedback":"..."} or {"decision":"revise","feedback":"..."}\n'
        f"Draft:\n{d}"
    )
    res = LLM_REVIEW.invoke(prompt)
    try:
        p = json.loads(res.content)
        return {"decision": p.get("decision","ok"), "feedback": p.get("feedback","Looks good.")}
    except Exception:
        return {"decision":"ok","feedback":"Looks good."}

@traced("rewrite")
def rewrite(state: QAState) -> QAState:
    d = state.get("draft","")
    fb = state.get("feedback","")
    res = LLM_ANSWER.invoke(f"Revise per feedback (do not add PII):\nFeedback:\n{fb}\nDraft:\n{d}")
    return {"draft": res.content}

@traced("finalize")
def finalize(_: QAState) -> QAState:
    return {}

def route_after_classify(state: QAState) -> Literal["finalize","sanitize"]:
    return "finalize" if state.get("decision") == "block" else "sanitize"

def route_after_review(state: QAState) -> Literal["finalize","rewrite"]:
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
graph.add_conditional_edges("classify", route_after_classify, {"finalize":"finalize","sanitize":"sanitize"})
graph.add_edge("sanitize", "answer")
graph.add_edge("answer", "review")
graph.add_conditional_edges("review", route_after_review, {"finalize":"finalize","rewrite":"rewrite"})
graph.add_edge("rewrite", "review")
graph.add_edge("finalize", END)

if __name__ == "__main__":
    start: QAState = {"query": "My email is john.smith@example.com. Draft a short intro to share with a client."}
    cfg = {"configurable": {"thread_id": "otel-moderatedqa-001"}}

    with SqliteSaver.from_conn_string("otel_moderatedqa.sqlite") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        final = app.invoke(start, config=cfg)
        print("\n--- FINAL ---")
        print("Decision:", final.get("decision"))
        print("Answer:\n", final.get("draft","<no draft>"))
