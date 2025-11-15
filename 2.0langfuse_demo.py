# demo_langfuse_v3_minimal.py
from __future__ import annotations
import os
from dotenv import load_dotenv

from openai import OpenAI
from langfuse import get_client

load_dotenv()

# Required env:
#   LANGFUSE_PUBLIC_KEY=...
#   LANGFUSE_SECRET_KEY=...
#   (optional) LANGFUSE_HOST=https://cloud.langfuse.com
#   OPENAI_API_KEY=...

lf = get_client()          # v3 client
oai = OpenAI()             # vanilla OpenAI client

def summarize(text: str) -> str:
    # Top-level span for the request
    with lf.start_as_current_span(
        name="summarize-request",
        input=text,
        metadata={"component": "demo", "kind": "summary"},
    ) as span:
        # Child "generation" for the LLM call (specialized span)
        with lf.start_as_current_generation(
            name="llm-summary",
            model="gpt-4o-mini",
            metadata={"purpose": "bullet-summary"},
        ) as gen:
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": f"Summarize in 3 bullets:\n{text}"},
                ],
            )
            output = resp.choices[0].message.content
            # Attach IO to the generation for Langfuse UI
            gen.update(input=text, output=output)

        # Attach final output to the parent span and optionally score it
        span.update(output=output)
        # span.score(name="helpfulness", value=1.0)   # optional

    lf.flush()  # ensure it’s sent before process exits
    return output

if __name__ == "__main__":
    print(summarize("Langfuse provides tracing and analytics for LLM apps, with spans and generations."))
