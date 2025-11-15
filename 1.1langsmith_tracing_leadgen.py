# demo1_leadgen_traced.py
from __future__ import annotations
import os, json, re, time
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai  # traces raw OpenAI calls
from langsmith import Client as LSClient
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
UA = os.getenv("USER_AGENT", "LeadGen-Demo/1.0 (+contact: trainer@example.com)")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set in environment/.env")

# --- LangSmith setup (optional but recommended) ---
# If you set LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY, LangChain calls are auto-traced.
# Wrapping OpenAI below ensures even your *manual* OpenAI calls are traced too.
ls_client = LSClient() if os.getenv("LANGSMITH_API_KEY") else None

# --- Models ---
# For manual OpenAI usage (RAG, scoring, email), traced via wrapper:
raw_openai = wrap_openai(OpenAI(api_key=OPENAI_API_KEY))

# --- Search helper ---
tavily = TavilySearchAPIWrapper()  # reads TAVILY_API_KEY

from urllib.parse import urlparse

def infer_company_from_url(url: str) -> str:
    """Fallback: infer a company-like name from the domain."""
    try:
        netloc = urlparse(url or "").netloc  # e.g. "www.acme.io"
        netloc = re.sub(r"^www\.", "", netloc)
        # take first label as a crude company name
        return netloc.split(".")[0].replace("-", " ").title()
    except Exception:
        return ""


def search_linkedin_people(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Use Tavily to find likely LinkedIn profile URLs by query."""
    q = f"site:linkedin.com/in {query}"
    hits = tavily.results(q, max_results=max_results) or []
    out = []
    for h in hits:
        url = h.get("url") or h.get("link")
        title = h.get("title") or ""
        snippet = h.get("content") or ""
        if url:
            out.append({"title": title, "url": url, "snippet": snippet})
    return out

def infer_name_title_company(snippet_or_title: str) -> Dict[str, str]:
    """Heuristic parse from search result text."""
    # Very lightweight heuristic; improves with an LLM call if needed.
    # Try patterns like: "Jane Doe - Head of Sales at Acme Corp | LinkedIn"
    name, title, company = "", "", ""
    t = snippet_or_title.strip()
    m = re.search(r"^([^|\-–]+)\s*[-–|]\s*([^@]+?)\s+at\s+(.+)$", t)
    if m:
        name = m.group(1).strip()
        title = m.group(2).strip()
        company = m.group(3).split("|")[0].strip()
    else:
        # Fall back: try " - Title | LinkedIn"
        m2 = re.search(r"^([^|\-–]+)\s*[-–|]\s*([^|]+)\|", t)
        if m2:
            name = m2.group(1).strip()
            title = m2.group(2).strip()
    return {"name": name, "title": title, "company": company}

def search_company_pages(company: str, max_urls: int = 3) -> List[str]:
    """Find official/about pages for the company."""
    urls: List[str] = []
    for q in [f"{company} official site", f"{company} about page", f"{company} product overview"]:
        hits = tavily.results(q, max_results=3) or []
        for h in hits:
            u = h.get("url") or h.get("link")
            if u and u not in urls:
                urls.append(u)
    return urls[:max_urls]

def load_text(url: str) -> str:
    """Fetch page text with a UA header. Avoids LinkedIn fetch; we fetch company site instead."""
    docs = WebBaseLoader(url, requests_kwargs={"headers": {"User-Agent": UA}}).load()
    return docs[0].page_content if docs else ""

def rag_answer(question: str, docs: List[str]) -> str:
    """Your snippet: Answer using ONLY provided docs; traced via wrap_openai(OpenAI())."""
    system_message = "Answer the user's question using only the provided information below:\n" + "\n\n".join(docs)
    resp = raw_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        temperature=0
    )
    return resp.choices[0].message.content

def score_lead(role_query: str, lead: Dict[str, str], company_summary: str) -> int:
    """LLM-based 1-10 score; traced via wrapped client."""
    prompt = f"""Role query: {role_query}

Lead:
name: {lead.get('name')}
title: {lead.get('title')}
company: {lead.get('company')}
url: {lead.get('url')}

Company summary (from RAG):
{company_summary}

Score suitability 1-10 for outreach (10=best). Return only an integer."""
    resp = raw_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    m = re.search(r"\d+", text)
    return max(1, min(10, int(m.group(0)))) if m else 5

def draft_outreach_email(lead: Dict[str, str], company_summary: str) -> str:
    """Short outreach draft; traced via wrapped client."""
    prompt = f"""Write a brief, friendly outreach email (90-120 words) to {lead.get('name') or 'the prospect'}.
Context:
- Prospect title: {lead.get('title')}
- Company: {lead.get('company')}
- Company summary: {company_summary}

Tone: concise, helpful, mention one concrete value prop, include a one-line CTA. No fluff."""
    resp = raw_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def leadgen_pipeline(role_query: str, max_leads: int = 3) -> List[Dict[str, Any]]:
    """End-to-end: search → parse → (optional) RAG company summary → score → email.
    RAG/score/email are called even if company context is missing, to ensure tracing.
    """
    results: List[Dict[str, Any]] = []

    # 1) Find potential LinkedIn leads (we don't fetch LinkedIn pages themselves)
    leads = search_linkedin_people(role_query, max_results=max_leads * 2)

    for hit in leads:
        parsed = infer_name_title_company(hit["title"] or hit["snippet"] or "")
        fallback_company = infer_company_from_url(hit.get("url", ""))

        lead = {
            "name": parsed.get("name") or (hit["title"] or "").split(" - ")[0],
            "title": parsed.get("title") or "",
            "company": parsed.get("company") or fallback_company,
            "url": hit.get("url", ""),
            "snippet": hit.get("snippet", ""),
        }

        # 2) Try to get company pages (about / product), fetch text, build small context
        docs: List[str] = []
        company_hint = lead["company"] or role_query  # always have *something* to search with

        for u in search_company_pages(company_hint, max_urls=3):
            try:
                txt = load_text(u)
                if txt:
                    docs.append(txt[:2000])  # small cap to keep prompt size reasonable
            except Exception:
                continue

        # 3) RAG: Summarize “What does this company/org do and who is the ICP?”
        if docs:
            question = f"What does {lead['company'] or company_hint} do? Who is the ideal customer profile?"
            summary = rag_answer(question, docs)
        else:
            # still call rag_answer so tracing happens; use a dummy doc
            summary = rag_answer(
                "Summarize this context for a sales lead generation demo.",
                ["No reliable external company context was found; respond with a generic placeholder summary."]
            )

        # 4) Score (will still be traced even if summary is generic)
        score = score_lead(role_query, lead, summary)

        # 5) Draft email (also traced)
        email = draft_outreach_email(lead, summary)

        result = {
            **lead,
            "company_summary": summary,
            "lead_score": score,
            "email_draft": email,
        }
        if not docs:
            result["note"] = "No real company context found; used generic summary."

        results.append(result)

        if len(results) >= max_leads:
            break

        # gentle pacing to avoid rate limits
        time.sleep(0.4)

    return results


# Optional: attach feedback to the most recent run (works when tracing is enabled)
def attach_feedback_example(run_id: Optional[str]):
    if ls_client and run_id:
        try:
            ls_client.create_feedback(run_id=run_id, key="review", score=1.0, comment="LeadGen demo looks good")
        except Exception:
            pass

# --------- Main function ----------
if __name__ == "__main__":
    role = "Head of Growth SaaS India"  # e.g., "VP Marketing fintech US"
    out = leadgen_pipeline(role, max_leads=3)
    print(json.dumps(out, indent=2, ensure_ascii=False))
