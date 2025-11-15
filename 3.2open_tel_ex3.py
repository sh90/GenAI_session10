from __future__ import annotations
import os, time, random
from typing import List, Dict
from dotenv import load_dotenv

# OTel
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from langchain_openai import ChatOpenAI

load_dotenv()

def init_tracer(service_name="Session10-OTel-RAG"):
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").rstrip("/")
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

tracer = init_tracer()
LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0)

def mock_retrieve(query: str) -> List[Dict]:
    # replace with your retriever; this simulates 2-4 docs
    time.sleep(0.2)
    n = random.randint(2,4)
    return [{"id": i, "text": f"Doc {i} about RAG eval best practices."} for i in range(n)]

def rag_answer(question: str) -> str:
    with tracer.start_as_current_span("retrieve") as s1:
        t0 = time.perf_counter()
        docs = mock_retrieve(question)
        s1.set_attribute("retrieved.count", len(docs))
        s1.set_attribute("retrieved.ids", ",".join(str(d["id"]) for d in docs))
        s1.set_attribute("latency.ms", round((time.perf_counter() - t0)*1000,2))

    with tracer.start_as_current_span("generate") as s2:
        ctx = "\n".join(d["text"] for d in docs)
        prompt = f"Use only this info:\n{ctx}\n\nQuestion: {question}\nAnswer in 5 sentences."
        t0 = time.perf_counter()
        res = LLM.invoke(prompt)
        s2.set_attribute("latency.ms", round((time.perf_counter() - t0)*1000,2))

        usage = getattr(res, "response_metadata", {}).get("token_usage", {})
        for k, v in usage.items():
            s2.set_attribute(f"llm.token_usage.{k}", v)
        answer = res.content

    with tracer.start_as_current_span("score") as s3:
        # toy scoring: penalize if "hallucination" appears; reward if "RAG" appears
        sc = 0
        sc += 1 if "RAG" in answer.upper() else 0
        sc -= 1 if "HALLUCIN" in answer.upper() else 0
        s3.set_attribute("score.simple", sc)

    return answer

if __name__ == "__main__":
    print(rag_answer("What are best practices for evaluating RAG in production?"))
