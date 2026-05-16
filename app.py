import os
import re
import warnings
import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Suppress ALL transformer/tokenizer noise BEFORE any imports ───────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

# Import the refactored backend logic
from backend.data_loader import load_staff, load_invigilation
from backend.router import route_query
from backend.search import _col, _series

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STAFF_PATH = os.path.join(BASE_DIR, "backend", "data", "ACSE Staff List.xlsx")
INVIG_PATH = os.path.join(BASE_DIR, "backend", "data", "Invigilation_Data.xlsx")

# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL HEAVY LIBS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

nlp      = _load_spacy()
embedder = _load_embedder()
SPACY_OK = nlp is not None
RAG_OK   = embedder is not None

from backend.llm import LLM_OK

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Agent for ACSE Department",
    page_icon="🤖",
    layout="wide",
)

CSS_PATH = os.path.join(BASE_DIR, "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ CSS file not found. App will load with default styles.")

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_ENV_KEY = os.getenv("ANTHROPIC_API_KEY", "")

for _k, _v in [
    ("chat_history",  []),
    ("df",            None),
    ("df_invig",      None),
    ("embeddings",    None),
    ("api_key",       _ENV_KEY),
    ("last_schedule", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

@st.cache_data(show_spinner=False)
def _cached_embeddings(filepath: str):
    if embedder is None:
        return None
    df = load_staff(filepath)
    texts = [" | ".join(str(v) for v in row.values if v) for _, row in df.iterrows()]
    return embedder.encode(texts, convert_to_numpy=True)

if st.session_state.df is None:
    if os.path.exists(STAFF_PATH):
        try:
            st.session_state.df = load_staff(STAFF_PATH)
            if RAG_OK:
                st.session_state.embeddings = _cached_embeddings(STAFF_PATH)
        except Exception as _e:
            st.error(f"❌ Failed to auto-load staff file: {_e}")
    else:
        st.warning(f"⚠️ Staff file not found at: `{STAFF_PATH}`")

if st.session_state.df_invig is None and os.path.exists(INVIG_PATH):
    try:
        st.session_state.df_invig = load_invigilation(INVIG_PATH)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
col_logo, col_title = st.columns([1, 10])
with col_title:
    st.markdown('<div class="page-title">ACSE AI Agent</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Intelligent Assistant for the Department of ACSE</p>', unsafe_allow_html=True)

st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

# Main Chat Area
if st.session_state.chat_history:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
        st.markdown(f'<div class="{cls}">{msg["html"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Action button row below chat
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
else:
    # Empty State
    st.markdown(
        "<div style='text-align:center;margin-top:60px;margin-bottom:40px;'>"
        "<div style='font-size:3rem; margin-bottom: 1rem;'>🤖</div>"
        "<h2 style='color:#1e293b; font-weight: 600;'>How can I help you today?</h2>"
        "<p style='color:#475569;font-size:1.1rem;'>"
        "Ask about the staff directory, invigilation schedules, or faculty replacements."
        "</p></div>",
        unsafe_allow_html=True,
    )
    
    # Quick Actions Grid
    st.markdown('<div class="quick-actions-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👨‍💼 Who is HoD?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "html": "Who is HoD?"})
            st.session_state.chat_history.append({"role": "bot",  "html": route_query("Who is HoD?", nlp, embedder, st.session_state.embeddings)})
            st.rerun()
    with col2:
        if st.button("👩‍🏫 List All Faculty", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "html": "List all faculty"})
            st.session_state.chat_history.append({"role": "bot",  "html": route_query("List all faculty", nlp, embedder, st.session_state.embeddings)})
            st.rerun()
    with col3:
        if st.button("📋 Generate Schedule", use_container_width=True):
            q_ = "Assign schedule for tomorrow's exam"
            st.session_state.chat_history.append({"role": "user", "html": q_})
            st.session_state.chat_history.append({"role": "bot",  "html": route_query(q_, nlp, embedder, st.session_state.embeddings)})
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([9, 1])
    with col1:
        user_input = st.text_input("q", placeholder="Type your question here…", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("↑")

if submitted and user_input.strip():
    raw = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "html": raw})
    reply = route_query(raw, nlp, embedder, st.session_state.embeddings)
    st.session_state.chat_history.append({"role": "bot", "html": reply})
    st.rerun()
