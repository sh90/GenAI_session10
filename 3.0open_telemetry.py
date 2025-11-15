

# Set up tracing with OpenTelemetry (a tracer + console exporter), so every step of the app is recorded as spans you can inspect.
# 
# Calls an LLM (OpenAI gpt-4o-mini) to generate a short greeting, wrapped inside a span so you get timing and metadata.
#
# Captures useful telemetry on the LLM call: model name, endpoint, latency (ms), token usage (if available), and any exceptions.
#
# Shows a tiny workflow: a parent span app.greet_user that calls a fake DB fetch db.fetch_user and then the LLM span llm.call—so you see parent/child relationships.
#
# Prints traces to the console (no external backend required) and prints the generated greeting.
#
# Easy to upgrade: swap the console exporter for OTLP to ship traces to Jaeger/Tempo/Datadog/Grafana later.
#
# Input: none (uses a hardcoded user_id and prompt).
# Output: a greeting string and structured span logs in your terminal.

from __future__ import annotations
import os, time
from dotenv import load_dotenv
from openai import OpenAI

# OpenTelemetry core + SDK (console exporter)
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.status import Status, StatusCode

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env file")

# ---- 1) Minimal tracer setup (console exporter) ----
resource = Resource.create({"service.name": "otel-llm-demo"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("demo.llm")

# ---- 2) LLM helper (wrap the model call in a span) ----
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def llm_answer(system_prompt: str, user_prompt: str) -> str:
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.vendor", "openai")
        span.set_attribute("llm.model", MODEL)
        span.set_attribute("llm.endpoint", "responses.create")

        t0 = time.perf_counter()
        try:
            # Simple, no special response_format, compatible with your earlier usage
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            dt = (time.perf_counter() - t0) * 1000
            span.set_attribute("llm.latency_ms", round(dt, 2))

            # token usage (fields can vary across SDK versions, handle defensively)
            usage = getattr(resp, "usage", None)
            if usage:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                          "input_tokens", "output_tokens"):
                    if hasattr(usage, k):
                        span.set_attribute(f"llm.usage.{k}", getattr(usage, k))

            # Try the new Responses API “output_text”, then fallbacks
            try:
                text = resp.output_text
            except Exception:
                # Fallback: dig into blocks
                text = ""
                blocks = getattr(resp, "output", [])
                if blocks and getattr(blocks[0], "content", []):
                    text = blocks[0].content[0].text

            if not text:
                span.add_event("llm.empty_output")
                text = "(no content returned)"
            return text

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

# ---- 3) A tiny “business flow” with parent/child spans ----
def fetch_user_profile(user_id: int) -> dict:
    with tracer.start_as_current_span("db.fetch_user") as span:
        span.set_attribute("db.system", "sqlite")
        span.set_attribute("db.operation", "SELECT")
        # pretend I/O
        time.sleep(0.03)
        profile = {"id": user_id, "name": "Aditi", "role": "Product Manager"}
        span.add_event("db.row_fetched", {"user_id": user_id})
        return profile

def greet_user(user_id: int) -> str:
    with tracer.start_as_current_span("app.greet_user") as span:
        profile = fetch_user_profile(user_id)
        span.set_attribute("user.id", profile["id"])
        span.set_attribute("user.name", profile["name"])

        system = "You are a concise, friendly assistant."
        user = f"Write one short sentence greeting {profile['name']} who is a {profile['role']}."
        text = llm_answer(system, user)
        span.add_event("llm.greeting_generated")
        return text

# ---- 4) Run it ----
if __name__ == "__main__":
    print(greet_user(42))
