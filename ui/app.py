"""Streamlit chat UI for the SBP Regulatory Assistant.

Requires the FastAPI backend running at http://localhost:8000.
Launch with: streamlit run ui/app.py
"""

import json
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1]))

API_BASE    = "http://localhost:8000"
CHUNKS_PATH = Path(__file__).parents[1] / "data" / "processed" / "raw_pages.jsonl"

EXAMPLE_QUERIES = [
    "What is the minimum capital requirement for a microfinance bank?",
    "What are the requirements for filing a Suspicious Transaction Report?",
    "What powers does the SBP Act grant to the State Bank of Pakistan?",
    "What prudential regulations apply to SME financing?",
    "What are the licensing requirements for a new banking company?",
]

DOC_TYPE_MAP = {
    "All Documents": None,
    "Laws": "laws",
    "Regulations": "regulations",
    "AML-CFT": "aml",
}

st.set_page_config(
    page_title="SBP Regulatory Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f7f8fa !important;
    color: #1a2332 !important;
}
.stApp {
    background-color: #f7f8fa !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e6ea !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] * { color: #1a2332 !important; }

/* ── Sidebar brand header ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 20px 0;
    border-bottom: 2px solid #006633;
    margin-bottom: 20px;
}
.brand-logo {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #006633, #004d26);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,102,51,0.25);
}
.brand-name {
    font-family: 'Merriweather', serif !important;
    font-size: 0.9rem;
    font-weight: 700;
    color: #006633 !important;
    line-height: 1.2;
    -webkit-text-fill-color: #006633 !important;
}
.brand-sub {
    font-size: 0.68rem;
    color: #6b7a8d !important;
    -webkit-text-fill-color: #6b7a8d !important;
    font-weight: 400;
    margin-top: 1px;
}

/* ── Status pill ── */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #edf7f0;
    border: 1px solid #a3d9b1;
    color: #1a6b35 !important;
    -webkit-text-fill-color: #1a6b35 !important;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 16px;
}
.status-offline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fef2f2;
    border: 1px solid #fca5a5;
    color: #991b1b !important;
    -webkit-text-fill-color: #991b1b !important;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 16px;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #16a34a;
    animation: blink 2s ease-in-out infinite;
}
.status-dot-off { background: #dc2626; animation: none; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Sidebar section labels ── */
.sidebar-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #9ba8b5 !important;
    -webkit-text-fill-color: #9ba8b5 !important;
    margin-bottom: 8px;
    margin-top: 4px;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #f7f8fa !important;
    border: 1px solid #d1d9e0 !important;
    border-radius: 8px !important;
    color: #1a2332 !important;
    font-size: 0.85rem !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #006633 !important;
    box-shadow: 0 0 0 3px rgba(0,102,51,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid #d1d9e0 !important;
    color: #374151 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    text-align: left !important;
    white-space: normal !important;
    line-height: 1.45 !important;
    padding: 10px 14px !important;
    height: auto !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.stButton > button:hover {
    background: #edf7f0 !important;
    border-color: #006633 !important;
    color: #006633 !important;
    box-shadow: 0 2px 8px rgba(0,102,51,0.14) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Clear button */
.clear-btn .stButton > button {
    background: #fff5f5 !important;
    border-color: #fca5a5 !important;
    color: #c53030 !important;
    text-align: center !important;
}
.clear-btn .stButton > button:hover {
    background: #fed7d7 !important;
    border-color: #fc8181 !important;
    color: #9b2c2c !important;
    box-shadow: 0 2px 8px rgba(197,48,48,0.12) !important;
}

/* ── Dividers ── */
hr { border: none !important; border-top: 1px solid #e9ecef !important; }

/* ── Main area top bar ── */
.topbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e6ea;
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 0 -1rem;
}
.topbar-title {
    font-family: 'Merriweather', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #006633;
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-meta {
    font-size: 0.75rem;
    color: #6b7a8d;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #006633 0%, #004d26 60%, #003d1f 100%);
    border-radius: 12px;
    padding: 36px 40px;
    margin: 20px 0 24px 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,102,51,0.2);
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 80px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(255,255,255,0.6);
    margin-bottom: 8px;
}
.hero-title {
    font-family: 'Merriweather', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
    line-height: 1.25;
}
.hero-sub {
    font-size: 0.875rem;
    color: rgba(255,255,255,0.75);
    max-width: 520px;
    line-height: 1.65;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #ffffff;
    margin-top: 16px;
}

/* ── Section heading ── */
.section-heading {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b7a8d;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2e6ea;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #e2e6ea !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    animation: slideUp 0.25s ease;
}
@keyframes slideUp {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}

/* User message — green left border */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid #006633 !important;
    background: #fafdfb !important;
}
/* Assistant — gold left border */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #b8960c !important;
    background: #fffdf5 !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #1a2332 !important;
    line-height: 1.75 !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatMessage"] code {
    background: #f0f4f0 !important;
    border: 1px solid #d4e6d4 !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    color: #006633 !important;
    font-size: 0.82rem !important;
}
[data-testid="stChatMessage"] strong { color: #0d1f0d !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid #c8d3dc !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #1a2332 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #006633 !important;
    box-shadow: 0 0 0 3px rgba(0,102,51,0.10), 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* ── Expander (Sources) ── */
[data-testid="stExpander"] {
    background: #f9fbf9 !important;
    border: 1px solid #d4e6d4 !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #006633 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* ── Source cards ── */
.source-card {
    background: #ffffff;
    border: 1px solid #e2ece2;
    border-left: 3px solid #006633;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.source-doc {
    font-size: 0.8rem;
    font-weight: 600;
    color: #004d26;
}
.source-meta {
    font-size: 0.72rem;
    color: #6b7a8d;
    margin-top: 2px;
}

/* ── Stats chips in topbar ── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #f0f7f2;
    border: 1px solid #c3dfc9;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #1a6b35;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f7f8fa; }
::-webkit-scrollbar-thumb { background: #c8d3dc; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #006633; }

/* ── Warnings / errors ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.84rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_document_list() -> dict[str, list[str]]:
    """Return distinct source_file values grouped by doc_type."""
    groups: dict[str, set[str]] = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    page = json.loads(line)
                    meta = page.get("metadata", {})
                    dt  = meta.get("doc_type", "other")
                    src = meta.get("source_file", "")
                    if src:
                        groups.setdefault(dt, set()).add(src)
                except json.JSONDecodeError:
                    continue
    return {k: sorted(v) for k, v in sorted(groups.items())}


@st.cache_data(show_spinner=False, ttl=30)
def get_api_health() -> dict | None:
    """Ping /health endpoint."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_query(question: str, doc_type: str | None) -> dict | None:
    """POST to /query and return the response dict, or None on error."""
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "doc_type": doc_type},
            timeout=120,
        )
        if resp.status_code == 429:
            st.warning("⏳ Groq rate limit reached (30 RPM free tier). Wait 2–3 seconds and try again.")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API at http://localhost:8000 — is `python api/run.py` running?")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again in a moment.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def submit_question(question: str, doc_type: str | None) -> None:
    """Append question, call API, append answer."""
    if not question.strip():
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Retrieving from regulatory documents…"):
        result = post_query(question, doc_type)
    if result:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": result.get("sources", []),
        })


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">🏛️</div>
        <div>
            <div class="brand-name">SBP Regulatory<br>Assistant</div>
            <div class="brand-sub">State Bank of Pakistan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Health
    health = get_api_health()
    if health and health.get("status") == "ok":
        total_pts = sum(
            c.get("points_count", 0) or 0
            for c in health.get("collections", [])
        )
        st.markdown(
            f'<div class="status-online"><span class="status-dot"></span>'
            f'API Connected &nbsp;·&nbsp; {total_pts:,} vectors</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-offline"><span class="status-dot status-dot-off"></span>'
            'API Offline</div>',
            unsafe_allow_html=True,
        )

    # Filter
    st.markdown('<div class="sidebar-label">Document Scope</div>', unsafe_allow_html=True)
    selected_filter = st.selectbox(
        label="scope",
        options=list(DOC_TYPE_MAP.keys()),
        index=0,
        label_visibility="collapsed",
    )
    active_doc_type = DOC_TYPE_MAP[selected_filter]

    st.divider()

    # Clear
    st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Document index
    st.markdown('<div class="sidebar-label">Document Library</div>', unsafe_allow_html=True)

    type_icons = {
        "law": "⚖️", "regulation": "📋",
        "aml": "🔎", "notification": "📢", "other": "📄"
    }
    doc_groups = load_document_list()
    if doc_groups:
        for group, docs in doc_groups.items():
            icon = type_icons.get(group, "📄")
            with st.expander(f"{icon} {group.title()}  ({len(docs)})", expanded=False):
                for doc in docs:
                    st.caption(f"· {doc}")
    else:
        st.caption("Document index unavailable.")

    st.divider()
    st.markdown(
        '<div style="font-size:0.67rem;color:#9ba8b5;text-align:center;padding-top:4px;">'
        'Groq · Qdrant · Cohere · Gemini</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main — Hero
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">State Bank of Pakistan · Regulatory Intelligence</div>
    <div class="hero-title">Regulatory Q&amp;A Assistant</div>
    <div class="hero-sub">
        Ask questions grounded in official SBP regulatory documents.
        Every answer is cited with the source document, section, and page number.
    </div>
    <div class="hero-badge">⚡ Powered by Hybrid RAG · BM25 + Vector Search + Cohere Rerank</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

st.markdown('<div class="section-heading">Suggested Queries</div>', unsafe_allow_html=True)

cols = st.columns(len(EXAMPLE_QUERIES))
for col, query in zip(cols, EXAMPLE_QUERIES):
    if col.button(query, use_container_width=True, help=query):
        submit_question(query, active_doc_type)
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 View Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    doc     = s.get("document", "Unknown")
                    section = s.get("section", "") or "—"
                    page    = s.get("page", "") or "—"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-doc">📄 {doc}</div>'
                        f'<div class="source-meta">Section: {section} &nbsp;·&nbsp; Page: {page}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about SBP regulations…"):
    submit_question(prompt, active_doc_type)
    st.rerun()
