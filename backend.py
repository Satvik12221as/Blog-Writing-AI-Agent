from __future__ import annotations

import operator
import os
import json
import re
from urllib import response
def safe_json_extract(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except:
        return None
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
# merge_content -> decide_images -> generate_and_place_images
# ==============================================================================


# ----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a blog planner.
Your job: decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts, well-known topics.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy/current events.

If needs_research=true:
- Provide 8–15 focused subtopic search queries covering different angles of the topic.
- For open_book weekly roundup, include queries reflecting the last 7 days.

Respond with ONLY a valid JSON object, no other text. Use this exact schema:
{"needs_research": true, "mode": "open_book", "reason": "Topic involves current events", "queries": ["query1", "query2"], "max_results_per_query": 5}
"""

def router_node(state: State) -> dict:
    import json
    import re
    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),

        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}")
    ]).content

    decision_data = safe_json_extract(response)

    if not decision_data:
        decision_data = {
            "needs_research": False,
            "mode": "closed_book",
            "reason": "Fallback due to invalid model output",
            "queries": [],
            "max_results_per_query": 5
        }

    decision_data.setdefault("needs_research", False)
    decision_data.setdefault("mode", "closed_book")
    decision_data.setdefault("reason", "Auto-filled reason")
    decision_data.setdefault("queries", [])
    decision_data.setdefault("max_results_per_query", 5)

    try:
        decision = RouterDecision(**decision_data)
    except Exception:
        # Final safety fallback
        decision = RouterDecision(
            needs_research=False,
            mode="closed_book",
            reason="Validation fallback",
            queries=[],
            max_results_per_query=5
        )

    # Recency logic.
    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------

def _iso_to_date(iso_str: Optional[str]) -> Optional[date]:
    """Convert ISO 'YYYY-MM-DD' string to date object, or return None."""
    if not iso_str:
        return None
    try:
        return date.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return None


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Perform web search using Tavily API.
    Returns empty list gracefully if API key is missing or package not installed.
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        print("[WARN] tavily-python not installed. Skipping web search. Run: pip install tavily-python")
        return []

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("[WARN] TAVILY_API_KEY not set. Skipping web search. Get a free key at https://tavily.com")
        return []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
    except Exception as e:
        print(f"[WARN] Tavily search failed for '{query}': {e}")
        return []

    results = []
    for result in response.get("results", []):
        results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "published_at": result.get("published_date", None),
            "snippet": result.get("content", ""),
            "source": result.get("source", "")
        })
    return results

RESEARCH_SYSTEM = """You are a research synthesizer.
Given raw web search results, produce a JSON object containing evidence items.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets concise but informative (2-3 sentences capturing the key facts).
- Deduplicate by URL.

Respond with ONLY a valid JSON object, no other text. Use this exact schema:
{"evidence": [{"title": "...", "url": "https://...", "published_at": "2025-01-15", "snippet": "...", "source": "Reuters"}]}
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []

    for q in queries:
        raw.extend(_tavily_search(q, max_results=5))

    if not raw:
        return {"evidence": []}

    response = llm.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            )
        ]
    ).content

    evidence_data = safe_json_extract(response)

    # 🔹 Fallback if Groq fails to produce valid JSON
    if not evidence_data or "evidence" not in evidence_data:
        return {"evidence": []}

    try:
        pack = EvidencePack(**evidence_data)
    except Exception:
        # If validation fails, skip research safely
        return {"evidence": []}

    # 🔹 Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    evidence = list(dedup.values())

    # 🔹 Apply recency filter for open_book mode
    if state.get("mode") == "open_book":
        try:
            as_of = date.fromisoformat(state["as_of"])
            cutoff = as_of - timedelta(days=int(state["recency_days"]))
            evidence = [
                e for e in evidence
                if (d := _iso_to_date(e.published_at)) and d >= cutoff
            ]
        except Exception:
            # Never crash on date issues
            pass

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior editorial planner for a popular online publication.
Produce an outline for a blog post that reads like a REAL, polished blog — not a textbook or encyclopedia entry.

Requirements:
- Create 5-8 tasks (sections), each with a compelling title, a clear goal, 3–6 detailed bullets, and a target_words count (200-600 per section).
- Section titles MUST be engaging and specific — NEVER use generic titles like "Overview", "Introduction", "Core Concepts", "Key Points", "Conclusion", "Summary".
  Good examples: "The Shifting Frontlines", "Why Silicon Valley Is Watching", "What Happens Next", "The Human Cost Nobody Talks About", "Breaking Down the Numbers".
- Bullets should be specific content directives, NOT generic placeholders like "Definition" or "Why it matters".
  Good bullet examples: "Open with the latest ceasefire breakdown in eastern Ukraine", "Compare current refugee numbers to 2022 peak", "Quote from UN Secretary-General's March statement".
- The tone field should match the topic: "conversational and direct" for general audiences, "analytical" for deep dives, "urgent and empathetic" for crisis topics.

Grounding:
- closed_book: evergreen topics, no evidence dependence.
- hybrid: use provided evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: news/current events roundup:
  - Set blog_kind="news_roundup"
  - Focus on events, analysis, and implications — NOT tutorials
  - If evidence is sparse, plan sections around general analysis of the topic rather than inventing specific events.

Respond with ONLY a valid JSON object, no other text. Use this schema:
{"blog_title": "Compelling Title Here", "audience": "General readers interested in...", "tone": "conversational and direct", "blog_kind": "explainer", "constraints": [], "tasks": [{"id": 1, "title": "Engaging Section Title", "goal": "One sentence goal", "bullets": ["Specific direction 1", "Specific direction 2", "Specific direction 3"], "target_words": 350, "tags": [], "requires_research": false, "requires_citations": false, "requires_code": false}]}
"""

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    response = llm.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n\n"
                    f"Evidence (ONLY use these for hybrid/open_book):\n"
                    f"{[e.model_dump() for e in evidence]}"
                )
            )
        ]
    ).content

    plan_data = safe_json_extract(response)

    # 🔹 Fallback if JSON extraction fails — produce a RICH plan, not bare bones
    topic = state["topic"]
    if not plan_data:
        plan_data = {
            "blog_title": f"{topic}: What You Need to Know",
            "audience": "General readers",
            "tone": "Conversational and insightful",
            "blog_kind": forced_kind or "explainer",
            "tasks": [
                {
                    "id": 1,
                    "title": "Setting the Scene",
                    "goal": "Hook the reader with the most compelling angle of this topic",
                    "bullets": [
                        f"Open with the most striking recent development related to {topic}",
                        "Paint a vivid picture of why readers should care right now",
                        "Preview what this article will cover"
                    ],
                    "target_words": 250,
                    "requires_research": mode != "closed_book",
                    "requires_citations": mode != "closed_book",
                    "requires_code": False,
                    "tags": ["intro"]
                },
                {
                    "id": 2,
                    "title": "The Bigger Picture",
                    "goal": "Put the topic in context — what led us here and what are the driving forces",
                    "bullets": [
                        f"Trace the key events or trends that shaped {topic}",
                        "Explain the underlying causes or motivations",
                        "Connect this to broader patterns the reader would recognize"
                    ],
                    "target_words": 400,
                    "requires_research": mode != "closed_book",
                    "requires_citations": mode != "closed_book",
                    "requires_code": False,
                    "tags": ["context"]
                },
                {
                    "id": 3,
                    "title": "What's Actually Happening",
                    "goal": "Break down the current state of affairs with specifics",
                    "bullets": [
                        "Present the most important facts and developments",
                        "Include relevant data points or statistics",
                        "Describe the key players involved and their positions",
                        "Highlight any surprising or underreported angles"
                    ],
                    "target_words": 500,
                    "requires_research": mode != "closed_book",
                    "requires_citations": mode != "closed_book",
                    "requires_code": False,
                    "tags": ["analysis"]
                },
                {
                    "id": 4,
                    "title": "The Human Side",
                    "goal": "Ground the topic in real human impact and experiences",
                    "bullets": [
                        "Describe how real people are affected",
                        "Include perspectives from different stakeholders",
                        "Use concrete examples or scenarios to make it tangible"
                    ],
                    "target_words": 350,
                    "requires_research": mode != "closed_book",
                    "requires_citations": mode != "closed_book",
                    "requires_code": False,
                    "tags": ["impact"]
                },
                {
                    "id": 5,
                    "title": "Where This Is Headed",
                    "goal": "Look at what comes next and what readers should watch for",
                    "bullets": [
                        "Outline the most likely scenarios going forward",
                        "Identify the key decision points or tipping points ahead",
                        "Leave the reader with a clear takeaway or call to attention"
                    ],
                    "target_words": 300,
                    "requires_research": False,
                    "requires_citations": False,
                    "requires_code": False,
                    "tags": ["outlook"]
                }
            ]
        }

    # 🔹 Ensure required keys exist (Groq safety)
    plan_data.setdefault("blog_title", f"{topic}: What You Need to Know")
    plan_data.setdefault("audience", "General readers")
    plan_data.setdefault("tone", "Conversational and insightful")
    plan_data.setdefault("blog_kind", "explainer")
    plan_data.setdefault("tasks", [])

    try:
        plan = Plan(**plan_data)
    except Exception:
        plan = Plan(
            blog_title=f"{topic}: What You Need to Know",
            audience="General readers",
            tone="Conversational and insightful",
            blog_kind=forced_kind or "explainer",
            tasks=[]
        )

    if not plan.tasks:
        plan.tasks = [
            Task(
                id=1,
                title="Setting the Scene",
                goal=f"Hook the reader with the most compelling angle of {topic}",
                bullets=[
                    f"Open with the most striking aspect of {topic}",
                    "Explain why this matters to the reader right now",
                    "Preview the key points this article will cover"
                ],
                target_words=400
            ),
            Task(
                id=2,
                title="Breaking It Down",
                goal="Explain the key aspects with depth and clarity",
                bullets=[
                    f"Analyze the most important dimensions of {topic}",
                    "Provide concrete examples and evidence",
                    "Connect the dots between different aspects"
                ],
                target_words=500
            ),
            Task(
                id=3,
                title="Looking Ahead",
                goal="Give the reader a clear sense of what to expect next",
                bullets=[
                    "Outline the trajectory and what to watch for",
                    "Provide a memorable closing thought"
                ],
                target_words=300
            )
        ]

    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}
# -------------------------
# 6) Fanout
# -------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a skilled journalist and blogger writing for a popular online publication.
Write ONE section of a blog post in Markdown.

Your writing MUST feel like a real blog post — engaging, human, and compelling. Follow these rules:

**Voice & Style:**
- Write like a skilled journalist or popular blogger, NOT a textbook or encyclopedia.
- Use vivid, concrete language. Show, don't just tell.
- Vary your sentence structure — mix short punchy sentences with longer flowing ones.
- Use storytelling where appropriate: anecdotes, scenarios, "imagine this" setups.
- NEVER write in a dictionary/definition format ("X refers to...", "X is defined as...").
- NEVER use bullet-point lists as your primary content format. Write in flowing paragraphs.
- Avoid corporate jargon and filler phrases like "it is important to note that", "in today's rapidly evolving landscape", "it is essential to understand".

**Structure:**
- Start with "## <Section Title>" — keep the title engaging.
- Cover ALL bullets from the plan, woven naturally into your prose.
- Target the specified word count (±15%).
- Use subheadings (###) sparingly — only if the section is 400+ words and benefits from visual breaks.

**For news/current events (blog_kind=="news_roundup"):**
- Focus on events, analysis, and real-world implications.
- Do NOT drift into tutorials, how-to guides, or technical walkthroughs.
- If citing evidence, use natural inline links: "according to [Reuters](URL)" or "[a recent report](URL) found that..."
- If no evidence is provided for a claim, write from general knowledge WITHOUT saying "Not found in provided sources". Instead, provide thoughtful analysis based on what is commonly known.

**For topics with evidence:**
- Weave source links naturally into sentences, e.g., "as [The Guardian reported](URL)..." or "according to [a UN assessment](URL)..."
- Do NOT dump all citations at the end. Integrate them.

**Code (only if requires_code==true):**
- Include at least one minimal, practical code snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    # Format evidence compactly to stay within token limits
    if evidence:
        evidence_parts = []
        for e in evidence[:8]:  # Cap at 8 sources to avoid token bloat
            snippet = (e.snippet or "")[:200]  # Truncate long snippets
            parts = [f"- {e.title} ({e.source or 'web'})"]
            if snippet:
                parts.append(f"  {snippet}")
            parts.append(f"  Link: {e.url}")
            evidence_parts.append("\n".join(parts))
        evidence_text = "\n".join(evidence_parts)
    else:
        evidence_text = "(No external sources available — write from general knowledge. Do NOT mention that sources are unavailable.)"

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {payload['topic']}\n"
                    f"As-of date: {payload.get('as_of')}\n\n"
                    f"--- YOUR SECTION ---\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"requires_code: {task.requires_code}\n\n"
                    f"Content directives (cover ALL of these naturally in your prose):{bullets_text}\n\n"
                    f"Available evidence:\n{evidence_text}\n\n"
                    f"Remember: Write engaging, flowing paragraphs. NO bullet-point lists as main content. "
                    f"NO dictionary-style definitions. Make it read like a real blog post.\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    response = llm.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n\n"
                    f"Merged markdown:\n{merged_md}"
                )
            ),
        ]
    ).content

    image_plan_data = safe_json_extract(response)

    # 🔹 If Groq fails to return valid JSON, skip images safely
    if not image_plan_data:
        return {
            "md_with_placeholders": merged_md,
            "image_specs": [],
        }

    # Ensure required keys exist
    image_plan_data.setdefault("md_with_placeholders", merged_md)
    image_plan_data.setdefault("images", [])

    try:
        image_plan = GlobalImagePlan(**image_plan_data)
    except Exception:
        # Final fallback safety
        return {
            "md_with_placeholders": merged_md,
            "image_specs": [],
        }

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout)
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app

