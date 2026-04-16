import os
import re
import json
import random
import warnings
import datetime

# ── Suppress ALL transformer/tokenizer noise BEFORE any imports ───────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  FILE PATHS  (auto-load — no manual upload needed)
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STAFF_PATH = os.path.join(BASE_DIR, "backend", "data", "ACSE Staff List.xlsx")
INVIG_PATH = os.path.join(BASE_DIR, "backend", "data", "Invigilation_Data.xlsx")

# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL HEAVY LIBS  ── cached so they load ONCE, not on every rerun
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

@st.cache_resource
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

try:
    import anthropic
    from sklearn.metrics.pairwise import cosine_similarity
    LLM_OK = True
except Exception:
    LLM_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Agent for ACSE Department",
    page_icon="🤖",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Inter:wght@400;500;600&display=swap');

html, body, .stApp {
    background-color: #f8fafc !important;
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
.main .block-container { padding-top: 2rem; max-width: 960px; }

.page-title {
    text-align: center;
    font-family: 'Sora', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    color: #1d4ed8;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.page-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 0.92rem;
    margin-top: 4px;
    margin-bottom: 0;
}
.page-divider {
    border: none;
    border-top: 1.5px solid #e2e8f0;
    margin: 1.2rem 0 1.5rem;
}
.sb-section {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 0.82rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.25rem 0 0.6rem;
}
.stack-row  { display:flex; align-items:center; gap:10px; padding:5px 0; }
.stack-dot-ok  { width:10px; height:10px; border-radius:50%; background:#22c55e; flex-shrink:0; }
.stack-dot-err { width:10px; height:10px; border-radius:50%; background:#ef4444; flex-shrink:0; }
.stack-label   { font-size:0.9rem; color:#334155; font-weight:500; }
.dir-info {
    background:#f1f5f9; border-radius:8px;
    padding:10px 14px; font-size:0.88rem;
    color:#475569; margin:6px 0;
}
.dir-info b { color:#1e293b; }
.chat-wrap { display:flex; flex-direction:column; gap:12px; padding-bottom:1.5rem; }
.bubble-user {
    align-self: flex-end;
    background: #1d4ed8;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    max-width: 72%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 1px 4px rgba(0,0,0,0.14);
}
.bubble-bot {
    align-self: flex-start;
    background: #ffffff;
    color: #1e293b;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 96%;
    font-size: 0.93rem;
    line-height: 1.55;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09);
    border: 1px solid #e2e8f0;
}
.staff-card {
    background: #f8fafc;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 8px;
}
.staff-card .sc-name {
    font-family:'Sora',sans-serif;
    font-weight:700; font-size:1.05rem;
    color:#1d4ed8; margin-bottom:8px;
}
.staff-card .sc-row   { font-size:0.88rem; color:#475569; margin:4px 0; }
.staff-card .sc-row b { color:#1e293b; }
.result-table {
    width:100%; border-collapse:collapse;
    font-size:0.86rem; margin-top:8px;
    border-radius:8px; overflow:hidden;
}
.result-table th {
    background:#1d4ed8; color:#fff;
    padding:8px 12px; text-align:left; font-weight:600;
}
.result-table td {
    padding:7px 12px;
    border-bottom:1px solid #e2e8f0; color:#334155;
}
.result-table tr:last-child td { border-bottom:none; }
.result-table tr:nth-child(even) td { background:#f1f5f9; }
.name-match {
    font-size:0.84rem; background:#eff6ff;
    border:1px solid #bfdbfe; border-radius:7px;
    padding:7px 10px; margin:3px 0; color:#1e40af;
}
.notify-card {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 4px solid #16a34a;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 8px;
}
.notify-card .nc-title {
    font-family:'Sora',sans-serif;
    font-weight:700; font-size:1rem;
    color:#15803d; margin-bottom:8px;
}
.notify-card .nc-row { font-size:0.88rem; color:#166534; margin:4px 0; }
.stTextInput input {
    border:1.5px solid #cbd5e1 !important;
    border-radius:12px !important;
    background:#ffffff !important;
    color:#1e293b !important;
    font-size:0.95rem !important;
}
.stTextInput input:focus {
    border-color:#2563eb !important;
    box-shadow:0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stFormSubmitButton button {
    background:#1d4ed8 !important; color:#fff !important;
    border-radius:10px !important; font-weight:600 !important;
    border:none !important; width:100% !important;
}
.stFormSubmitButton button:hover { background:#1e40af !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

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

# ══════════════════════════════════════════════════════════════════════════════
#  STAFF LOADER
# ══════════════════════════════════════════════════════════════════════════════
def find_header_row(raw_df: pd.DataFrame) -> int:
    for i, row in raw_df.iterrows():
        vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        if any(k in vals for k in ("name", "designation", "phone", "ecode", "email",
                                    "name of the faculty")):
            return i
        joined = " ".join(vals)
        if "name" in joined and ("design" in joined or "phone" in joined):
            return i
    return 0


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename duplicate column names so every column is unique."""
    seen: dict = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


def load_staff(filepath) -> pd.DataFrame:
    raw  = pd.read_excel(filepath, header=None)
    hrow = find_header_row(raw)
    df   = pd.read_excel(filepath, header=hrow)
    df.columns = [str(c).strip().title() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)

    # Drop entirely-empty columns
    df = df.dropna(axis=1, how="all")

    # Rename unnamed columns using positional heuristics
    known_order = [
        "Sl.No", "E.Code", "Name", "Designation", "Mobile No",
        "Qualification", "Year Of Passing", "University",
        "Ph.D", "Ph.D Year", "Ph.D University",
        "Date Of Joining", "Experience", "Pay Scale", "Basic Pay"
    ]
    unnamed_idx = 0
    new_cols = []
    for col in df.columns:
        if str(col).lower().startswith("unnamed"):
            if unnamed_idx < len(known_order):
                new_cols.append(known_order[unnamed_idx])
            else:
                new_cols.append(f"Extra_{unnamed_idx}")
            unnamed_idx += 1
        else:
            new_cols.append(col)
    df.columns = new_cols

    # Remove mostly-empty Extra_ columns
    df = df[[c for c in df.columns if not (
        str(c).startswith("Extra_") and
        df[c].astype(str).str.strip().isin(["", "nan", "none"]).mean() > 0.8
    )]]

    if "Name" in df.columns:
        df = df[df["Name"].notna()]
        df = df[~df["Name"].astype(str).str.strip().str.lower().isin(["nan", "none", ""])]

    for col in df.columns:
        if "name" in col.lower() and col != "Name":
            df = df.rename(columns={col: "Name"})
            break

    df = df.fillna("")

    # ── FIX: deduplicate columns (merged/wrapped Excel cells cause duplicates) ─
    df = _dedup_columns(df)

    return df


def load_invigilation(filepath) -> pd.DataFrame:
    try:
        raw  = pd.read_excel(filepath, header=None)
        hrow = find_header_row(raw)
        df   = pd.read_excel(filepath, header=hrow)
        df.columns = [str(c).strip().title() for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)
        df = df.fillna("")
        df = _dedup_columns(df)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _cached_embeddings(filepath: str):
    """Build staff embeddings once and cache on disk key = filepath."""
    if embedder is None:
        return None
    df = load_staff(filepath)
    texts = [" | ".join(str(v) for v in row.values if v) for _, row in df.iterrows()]
    return embedder.encode(texts, convert_to_numpy=True)


# ── Auto-load on first run ────────────────────────────────────────────────────
if st.session_state.df is None:
    if os.path.exists(STAFF_PATH):
        try:
            _df = load_staff(STAFF_PATH)
            st.session_state.df = _df
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
#  SAFE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _get(row, *keys) -> str:
    for k in keys:
        for col in row.index:
            if col.lower().replace(" ", "").replace(".", "") == k.lower().replace(" ", "").replace(".", ""):
                v = str(row[col]).strip()
                if v and v.lower() not in ("nan", "none", ""):
                    return v
    return "—"


def _col(df, *keys):
    """Return first column name from df whose name contains any of the given keywords."""
    for k in keys:
        for col in df.columns:
            if k.lower() in col.lower():
                return col
    return None


def _series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Safely extract a column as a Series.
    If duplicate column names exist, pandas returns a DataFrame — this always
    returns a proper Series (taking the first occurrence).
    """
    data = df[col_name]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data.astype(str)

# ══════════════════════════════════════════════════════════════════════════════
#  LLM QUERY PARSER
# ══════════════════════════════════════════════════════════════════════════════
def llm_parse_query(query: str, api_key: str) -> dict:
    _default = {"intent": "search", "person": None, "role": None,
                "room": None, "floor": None, "date": None, "action": None}
    if not LLM_OK or not api_key:
        return _default
    system = (
        "You are an NLP assistant for a college staff directory and invigilation system. "
        "Given a user query, extract: "
        "intent (search / list / greeting / invigilation / schedule / replace / notify / unknown), "
        "person name if mentioned, "
        "role/designation keyword if mentioned, "
        "room number if mentioned, "
        "floor number if mentioned, "
        "date if mentioned (ISO format or 'tomorrow'), "
        "action keyword if mentioned (assign/replace/notify/schedule). "
        'Return ONLY compact JSON: {"intent":"search","person":null,"role":null,'
        '"room":null,"floor":null,"date":null,"action":null}. '
        "Use null for missing fields. No markdown, no explanation."
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp   = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        parsed = json.loads(resp.content[0].text.strip())
        for k in ("person", "role", "room", "floor", "date", "action"):
            if isinstance(parsed.get(k), str) and parsed[k].strip().lower() in ("null", "none", ""):
                parsed[k] = None
        return parsed
    except Exception:
        return _default

# ══════════════════════════════════════════════════════════════════════════════
#  STAFF SEARCH
# ══════════════════════════════════════════════════════════════════════════════
def search_staff(df: pd.DataFrame, parsed: dict, raw_query: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    q        = raw_query.lower().strip()
    desg_col = _col(df, "desig")
    name_col = _col(df, "name")

    def _desg_filter(pattern):
        if not desg_col:
            return pd.DataFrame()
        # Use _series() to guard against duplicate-column DataFrames
        mask = _series(df, desg_col).str.contains(pattern, flags=re.IGNORECASE, regex=True)
        return df[mask]

    # ── Greetings ─────────────────────────────────────────────────────────────
    if parsed.get("intent") == "greeting":
        return pd.DataFrame()

    # ── HoD ───────────────────────────────────────────────────────────────────
    if re.search(r"\bhod\b", q) or "head of department" in q or "head of dept" in q:
        if desg_col:
            col_vals = _series(df, desg_col)            # ← safe Series
            mask_hod = col_vals.str.contains(r"(?i)\bhod\b", regex=True)
            mask_dean_no_hod = (
                col_vals.str.contains(r"(?i)dean", regex=True) & ~mask_hod
            )
            hits = df[mask_hod & ~mask_dean_no_hod]
            if not hits.empty:
                return hits
            hits = df[mask_hod]
            if not hits.empty:
                return hits

    # ── Dean ──────────────────────────────────────────────────────────────────
    if re.search(r"\bdean\b", q):
        hits = _desg_filter(r"\bdean\b")
        if not hits.empty:
            return hits

    # ── Principal ─────────────────────────────────────────────────────────────
    if re.search(r"\bprincipal\b", q):
        hits = _desg_filter(r"\bprincipal\b")
        if not hits.empty:
            return hits

    # ── Professor tiers (most specific first) ─────────────────────────────────
    tier_map = [
        ("asst. prof. senior level", r"asst\.?\s*prof\.?\s*senior"),
        ("senior level",              r"senior\s*level"),
        ("associate professor",       r"associate\s*professor"),
        ("assistant professor",       r"assistant\s*professor"),
        ("asst professor",            r"asst\.?\s*professor"),
        ("asst prof",                 r"asst\.?\s*prof"),
        ("professor",                 r"\bprofessor\b"),
        ("lecturer",                  r"\blecturer\b"),
        ("lab technician",            r"lab\s*technician"),
        ("attender",                  r"\battender\b"),
    ]
    for kw, pattern in tier_map:
        if kw in q:
            hits = _desg_filter(pattern)
            if not hits.empty:
                return hits

    # ── All faculty / staff ───────────────────────────────────────────────────
    if any(x in q for x in ("all faculty", "all staff", "list all", "show all", "full list")):
        return df

    # ── Named person from LLM ─────────────────────────────────────────────────
    person = parsed.get("person")
    if person and name_col:
        mask = _series(df, name_col).str.contains(
            re.escape(person), flags=re.IGNORECASE, regex=True)
        if df[mask].shape[0] > 0:
            return df[mask]

    # ── Role from LLM ─────────────────────────────────────────────────────────
    role = parsed.get("role")
    if role and desg_col:
        hits = _desg_filter(re.escape(role))
        if not hits.empty:
            return hits

    # ── RAG semantic search ───────────────────────────────────────────────────
    if embedder is not None and st.session_state.embeddings is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        q_emb = embedder.encode([raw_query], convert_to_numpy=True)
        sims  = cosine_similarity(q_emb, st.session_state.embeddings)[0]
        top   = np.argsort(sims)[::-1][:3]
        if sims[top[0]] > 0.35:
            return df.iloc[top]

    # ── spaCy PERSON entity ───────────────────────────────────────────────────
    if nlp is not None and name_col:
        doc = nlp(raw_query)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                mask = _series(df, name_col).str.contains(
                    re.escape(ent.text), flags=re.IGNORECASE, regex=True)
                if df[mask].shape[0] > 0:
                    return df[mask]

    # ── Direct name keyword match ─────────────────────────────────────────────
    if name_col:
        name_series = _series(df, name_col)
        for _, row in df.iterrows():
            name_val = str(row[name_col]).strip()
            if len(name_val) > 3 and name_val.lower() in q:
                return df[name_series.str.lower() == name_val.lower()]

    # ── Keyword fallback ──────────────────────────────────────────────────────
    words = [w for w in re.split(r"\s+", raw_query) if len(w) > 3]
    if words:
        combined = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include="object").columns:
            for w in words:
                combined |= _series(df, col).str.contains(
                    re.escape(w), flags=re.IGNORECASE, regex=True)
        hits = df[combined]
        if not hits.empty:
            return hits

    return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
#  INVIGILATION SEARCH
# ══════════════════════════════════════════════════════════════════════════════
def search_invigilation(df_invig: pd.DataFrame, raw_query: str, parsed: dict) -> str:
    if df_invig is None or df_invig.empty:
        return "<p style='color:#ef4444;'>❌ Invigilation data not loaded. Check backend/data/Invigilation_Data.xlsx</p>"

    q = raw_query.lower().strip()

    room_col    = _col(df_invig, "room")
    floor_col   = _col(df_invig, "floor")
    name_col    = _col(df_invig, "name", "faculty")
    date_col    = _col(df_invig, "date")

    # ── Room-based query ──────────────────────────────────────────────────────
    room_match = re.search(r'\b(\d{3,4})\b', q)
    if room_match:
        room_no = room_match.group(1)
        hits = df_invig.copy()
        if room_col:
            hits = hits[_series(hits, room_col).str.contains(room_no, na=False)]
        date_str = parsed.get("date")
        if date_str and date_col:
            target = (
                (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
                if date_str == "tomorrow" else date_str
            )
            hits = hits[_series(hits, date_col).str.contains(target, na=False)]
        if hits.empty:
            return f"<p style='color:#ef4444;'>❌ No invigilation found for room <b>{room_no}</b>.</p>"
        return _invig_table_html(hits, f"Room {room_no} Invigilation")

    # ── Floor-based query ─────────────────────────────────────────────────────
    floor_match = re.search(r'floor\s*(\d+)|(\d+)\s*(st|nd|rd|th)?\s*floor', q)
    if floor_match and floor_col:
        floor_no = floor_match.group(1) or floor_match.group(2)
        hits = df_invig[_series(df_invig, floor_col).str.contains(floor_no, na=False)]
        if hits.empty:
            return f"<p style='color:#ef4444;'>❌ No invigilation found for floor {floor_no}.</p>"
        return _invig_table_html(hits, f"Floor {floor_no} Invigilation")

    # ── Faculty-specific timetable ────────────────────────────────────────────
    if name_col and st.session_state.df is not None:
        staff_df  = st.session_state.df
        sname_col = _col(staff_df, "name")
        if sname_col:
            for _, srow in staff_df.iterrows():
                faculty_name = str(srow[sname_col]).strip()
                parts = faculty_name.split()
                for part in parts:
                    if len(part) > 3 and part.lower() in q:
                        hits = df_invig[
                            _series(df_invig, name_col).str.contains(
                                re.escape(faculty_name), flags=re.IGNORECASE, na=False)
                        ]
                        if hits.empty:
                            hits = df_invig[
                                _series(df_invig, name_col).str.contains(
                                    re.escape(part), flags=re.IGNORECASE, na=False)
                            ]
                        if not hits.empty:
                            return _invig_table_html(hits, f"Invigilation Schedule: {faculty_name}")
                        break

    # ── Date-based query ──────────────────────────────────────────────────────
    date_str = parsed.get("date")
    if date_str and date_col:
        target = (
            (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
            if date_str == "tomorrow" else date_str
        )
        hits = df_invig[_series(df_invig, date_col).str.contains(target, na=False)]
        if not hits.empty:
            return _invig_table_html(hits, f"Invigilation on {target}")

    return _invig_table_html(df_invig, "Full Invigilation Schedule")


def _invig_table_html(df: pd.DataFrame, title: str) -> str:
    n = len(df)
    if n == 0:
        return "<p style='color:#ef4444;'>❌ No records found.</p>"

    display_cols = [c for c in df.columns
                    if str(c).strip() and "unnamed" not in c.lower()][:8]

    headers   = "".join(f"<th>{c}</th>" for c in display_cols)
    rows_html = "".join(
        "<tr>" + "".join(
            f"<td>{str(row[c]) if str(row[c]).lower() not in ('nan','none','') else '—'}</td>"
            for c in display_cols
        ) + "</tr>"
        for _, row in df.iterrows()
    )
    return f"""
<p style='color:#1d4ed8;font-weight:700;font-size:1rem;margin-bottom:6px;'>📋 {title} ({n} record(s))</p>
<table class="result-table">
  <thead><tr>{headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""

# ══════════════════════════════════════════════════════════════════════════════
#  SCHEDULE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_schedule(df: pd.DataFrame, raw_query: str, parsed: dict) -> str:
    if df is None or df.empty:
        return "<p style='color:#ef4444;'>❌ Staff data not loaded.</p>"

    date_str = parsed.get("date")
    if date_str == "tomorrow" or "tomorrow" in raw_query.lower():
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%d-%m-%Y")
    elif date_str:
        target_date = date_str
    else:
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%d-%m-%Y")

    room_matches = re.findall(r'\b(\d{3,4})\b', raw_query)
    rooms = room_matches if room_matches else [
        "301", "302", "303", "401", "402", "403",
        "501", "502", "601", "602", "603", "607"
    ]

    name_col = _col(df, "name")
    desg_col = _col(df, "desig")
    if not name_col:
        return "<p style='color:#ef4444;'>❌ Name column not found in staff data.</p>"

    exclude_roles = ["attender", "lab technician", "peon", "clerk"]
    faculty_df = df.copy()
    if desg_col:
        for role in exclude_roles:
            faculty_df = faculty_df[
                ~_series(faculty_df, desg_col).str.lower().str.contains(role)
            ]

    faculty_names = _series(faculty_df, name_col).tolist()
    faculty_names = [f for f in faculty_names if f.strip() and f.lower() not in ("nan", "none", "")]

    if not faculty_names:
        return "<p style='color:#ef4444;'>❌ No faculty available for scheduling.</p>"

    sessions = ["FN (9:00 AM - 12:00 PM)", "AN (2:00 PM - 5:00 PM)"]
    schedule = []
    idx = 0
    for session in sessions:
        random.shuffle(faculty_names)
        for room in rooms:
            faculty = faculty_names[idx % len(faculty_names)]
            schedule.append({
                "Date": target_date,
                "Session": session,
                "Room No": room,
                "Faculty Assigned": faculty,
                "Status": "Scheduled"
            })
            idx += 1

    schedule_df = pd.DataFrame(schedule)
    st.session_state.last_schedule = schedule_df

    headers   = "".join(f"<th>{c}</th>" for c in schedule_df.columns)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{row[c]}</td>" for c in schedule_df.columns) + "</tr>"
        for _, row in schedule_df.iterrows()
    )
    return f"""
<p style='color:#16a34a;font-weight:700;font-size:1rem;margin-bottom:6px;'>
✅ Invigilation Schedule Generated for <b>{target_date}</b> ({len(schedule)} assignments)</p>
<table class="result-table">
  <thead><tr>{headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p style='color:#64748b;font-size:0.82rem;margin-top:8px;'>
💡 To replace a faculty, type: <i>"Replace [Faculty Name] with available faculty"</i></p>"""

# ══════════════════════════════════════════════════════════════════════════════
#  FACULTY REPLACEMENT
# ══════════════════════════════════════════════════════════════════════════════
def replace_faculty(df: pd.DataFrame, raw_query: str, parsed: dict) -> str:
    if df is None or df.empty:
        return "<p style='color:#ef4444;'>❌ Staff data not loaded.</p>"

    schedule_df = st.session_state.get("last_schedule")
    if schedule_df is None or schedule_df.empty:
        return ("<p style='color:#f59e0b;'>⚠️ No active schedule found. "
                "Please generate a schedule first by asking: "
                "<i>'Assign schedule for tomorrow's exam'</i></p>")

    name_col = _col(df, "name")
    if not name_col:
        return "<p style='color:#ef4444;'>❌ Name column not found.</p>"

    absent_faculty = parsed.get("person")
    if not absent_faculty:
        m = re.search(r'replace\s+([A-Za-z\s\.]+?)(?:\s+with|\s+due|\s+because|$)',
                      raw_query, re.IGNORECASE)
        if m:
            absent_faculty = m.group(1).strip()
        else:
            m2 = re.search(r'([A-Za-z\s\.]+?)\s+(?:is\s+)?on\s+leave', raw_query, re.IGNORECASE)
            if m2:
                absent_faculty = m2.group(1).strip()

    if not absent_faculty:
        return ("<p style='color:#f59e0b;'>⚠️ Please mention the faculty name to replace. "
                "Example: <i>'Replace Dr. Kumar with available faculty'</i></p>")

    mask = schedule_df["Faculty Assigned"].str.contains(
        re.escape(absent_faculty), flags=re.IGNORECASE, regex=True)
    if not mask.any():
        return f"<p style='color:#ef4444;'>❌ <b>{absent_faculty}</b> not found in current schedule.</p>"

    assigned_faculty = set(schedule_df["Faculty Assigned"].tolist())
    all_faculty = [
        str(r[name_col]).strip() for _, r in df.iterrows()
        if str(r[name_col]).strip() and str(r[name_col]).strip().lower() not in ("nan", "none", "")
    ]
    available = [f for f in all_faculty if f not in assigned_faculty]
    if not available:
        available = [f for f in all_faculty
                     if not re.search(re.escape(absent_faculty), f, re.IGNORECASE)]
    if not available:
        return "<p style='color:#ef4444;'>❌ No available replacement faculty found.</p>"

    replacement = random.choice(available)

    updated_rows = []
    for _, row in schedule_df[mask].iterrows():
        updated_rows.append({
            "Date": row["Date"],
            "Session": row["Session"],
            "Room No": row["Room No"],
            "Original Faculty": absent_faculty,
            "Replacement Faculty": replacement,
            "Status": "Replaced ✅"
        })

    schedule_df.loc[mask, "Faculty Assigned"] = replacement
    schedule_df.loc[mask, "Status"] = "Replaced"
    st.session_state.last_schedule = schedule_df

    rep_df = pd.DataFrame(updated_rows)

    notif_html = f"""
<div class="notify-card">
  <div class="nc-title">📲 WhatsApp Notification — Sent to Replacement Faculty Only</div>
  <div class="nc-row">👤 <b>Notified Faculty :</b> {replacement}</div>
  <div class="nc-row">📋 <b>Assigned Rooms  :</b> {', '.join(rep_df['Room No'].tolist())}</div>
  <div class="nc-row">📅 <b>Date            :</b> {rep_df['Date'].iloc[0]}</div>
  <div class="nc-row">⏰ <b>Sessions        :</b> {', '.join(rep_df['Session'].tolist())}</div>
  <div class="nc-row" style="margin-top:8px;font-style:italic;color:#166534;">
    ✉️ Message: "Dear {replacement}, you have been assigned as invigilation replacement for
    <b>{absent_faculty}</b> on {rep_df['Date'].iloc[0]}. Please report to Room(s)
    {', '.join(rep_df['Room No'].tolist())}. Thank you."
  </div>
</div>"""

    headers   = "".join(f"<th>{c}</th>" for c in rep_df.columns)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{row[c]}</td>" for c in rep_df.columns) + "</tr>"
        for _, row in rep_df.iterrows()
    )
    table_html = f"""
<p style='color:#1d4ed8;font-weight:700;margin-bottom:6px;'>
🔄 Replacement Summary for <b>{absent_faculty}</b></p>
<table class="result-table">
  <thead><tr>{headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""

    return table_html + notif_html

# ══════════════════════════════════════════════════════════════════════════════
#  HTML RENDERERS for staff results
# ══════════════════════════════════════════════════════════════════════════════
def results_to_html(results: pd.DataFrame, is_detail: bool = False) -> str:
    n = len(results)

    if n == 0:
        return ("<p style='color:#ef4444;margin:0;font-weight:600;'>"
                "❌ No matching staff found for your query.</p>")

    count_html = (f"<p style='color:#16a34a;font-weight:600;margin:0 0 8px;'>"
                  f"✅ Found {n} record(s)</p>")

    if n == 1 or is_detail:
        row   = results.iloc[0]
        name  = _get(row, "Name")
        desg  = _get(row, "Designation", "Desig")
        phone = _get(row, "Mobile No", "Phone", "Mobile", "Contact",
                     "Mobileno", "Mobile No.", "Phoneno", "Phone No")
        ecode = _get(row, "E.Code", "Ecode", "ECode", "EmployeeCode", "Employee Code", "Emp Code")
        email = _get(row, "Email", "Mail", "Email Id", "Email Address")

        extra_rows = ""
        skip_keys  = {"name", "designation", "desig", "mobileno", "mobileno.",
                      "mobile", "phone", "phoneno", "phoneno.", "ecode", "email",
                      "mail", "emailid", "emailaddress"}
        icon_map = {
            "sl": "🔢", "slno": "🔢", "emp": "🆔", "empcode": "🆔",
            "qualification": "🎓", "qual": "🎓",
            "dateofjoining": "📅", "doj": "📅", "joining": "📅",
            "department": "🏢", "dept": "🏢",
            "experience": "⏳", "exp": "⏳",
            "specialization": "🔬", "special": "🔬",
            "university": "🏛️", "college": "🏛️",
        }
        for col in row.index:
            col_str = str(col).strip()
            if not col_str or col_str.lower().startswith("unnamed") or col_str.lower() == "index":
                continue
            ck = col_str.lower().replace(" ", "").replace(".", "").replace("_", "")
            if ck in skip_keys:
                continue
            v = str(row[col]).strip()
            if not v or v.lower() in ("nan", "none", ""):
                continue
            icon = "📌"
            for key, ic in icon_map.items():
                if key in ck:
                    icon = ic
                    break
            extra_rows += f'<div class="sc-row">{icon} <b>{col_str} :</b> {v}</div>'

        card = f"""
<div class="staff-card">
  <div class="sc-name">👤 {name}</div>
  <div class="sc-row">🏷️ <b>Designation :</b> {desg}</div>
  <div class="sc-row">📞 <b>Phone       :</b> {phone}</div>
  <div class="sc-row">📧 <b>Email       :</b> {email}</div>
  <div class="sc-row">🆔 <b>E.Code      :</b> {ecode}</div>
  {extra_rows}
</div>"""
        return count_html + card

    # Multiple results → table
    display_cols = [c for c in results.columns
                    if str(c).strip() and "unnamed" not in c.lower()][:7]
    headers   = "".join(f"<th>{c}</th>" for c in display_cols)
    rows_html = "".join(
        "<tr>" + "".join(
            f"<td>{str(row[c]) if str(row[c]).lower() not in ('nan','none') else '—'}</td>"
            for c in display_cols
        ) + "</tr>"
        for _, row in results.iterrows()
    )
    return count_html + f"""
<table class="result-table">
  <thead><tr>{headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""

# ══════════════════════════════════════════════════════════════════════════════
#  INTENT ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def route_query(raw: str) -> str:
    q        = raw.lower().strip()
    df       = st.session_state.df
    df_invig = st.session_state.df_invig
    api_key  = st.session_state.api_key

    _greetings = ("hi", "hello", "hey", "good morning", "good afternoon", "good evening", "namaste")
    if any(q.startswith(g) for g in _greetings):
        return ("👋 Hello! I'm the ACSE AI Agent. I can help you with:<br>"
                "• Staff info (HoD, professors, phone numbers)<br>"
                "• Invigilation schedules by room / floor / faculty<br>"
                "• Auto-generate exam invigilation schedule<br>"
                "• Faculty replacement with WhatsApp notification<br>"
                "Ask me anything!")

    if df is None:
        return "⚠️ Staff data not loaded. Please check the <b>backend/data/</b> folder."

    parsed = llm_parse_query(raw, api_key)
    intent = parsed.get("intent", "search")

    is_detail = any(x in q for x in ("details", "detail", "full info",
                                      "phone number", "contact", "mobile", "email",
                                      "all info", "information about"))

    # Schedule
    sched_keywords = ("assign schedule", "generate schedule", "create schedule",
                      "schedule for tomorrow", "schedule for today", "assign invig",
                      "generate invig", "create invig", "tomorrow's exam", "exam schedule",
                      "assign rooms", "invigilation schedule")
    if intent == "schedule" or any(kw in q for kw in sched_keywords):
        return generate_schedule(df, raw, parsed)

    # Replacement
    replace_keywords = ("replace", "on leave", "absent", "substitute", "swap faculty")
    if intent == "replace" or any(kw in q for kw in replace_keywords):
        return replace_faculty(df, raw, parsed)

    # Notify
    if intent == "notify" or "notify" in q or "notification" in q or "whatsapp" in q:
        if st.session_state.last_schedule is not None:
            return replace_faculty(df, raw, parsed)
        return "<p style='color:#f59e0b;'>⚠️ No replacement found. Generate a schedule and assign a replacement first.</p>"

    # Invigilation
    invig_keywords = ("invig", "room", "floor", "exam duty", "exam schedule",
                      "who is in room", "which faculty", "assigned to room",
                      "timetable", "duty chart")
    if intent == "invigilation" or any(kw in q for kw in invig_keywords):
        return search_invigilation(df_invig, raw, parsed)

    # Staff search
    results = search_staff(df, parsed, raw)
    return results_to_html(results, is_detail=(is_detail and len(results) == 1))

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown('<div class="sb-section">🤖 AI Stack Status</div>', unsafe_allow_html=True)
    llm_ready    = LLM_OK and bool(st.session_state.api_key)
    invig_loaded = st.session_state.df_invig is not None and not st.session_state.df_invig.empty
    for label, ok in [
        ("LLM (Claude)",      llm_ready),
        ("RAG (Embeddings)",  RAG_OK),
        ("NLP (spaCy)",       SPACY_OK),
        ("Invigilation Data", invig_loaded),
    ]:
        dot = "stack-dot-ok" if ok else "stack-dot-err"
        st.markdown(
            f'<div class="stack-row"><div class="{dot}"></div>'
            f'<span class="stack-label">{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="sb-section">🔑 Anthropic API Key</div>', unsafe_allow_html=True)
    st.caption("Enter key to enable LLM")
    key_input = st.text_input(
        "api_key_field",
        value=st.session_state.api_key,
        type="password",
        label_visibility="collapsed",
    )
    if key_input != st.session_state.api_key:
        st.session_state.api_key = key_input
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sb-section">👥 Staff Directory</div>', unsafe_allow_html=True)
    if st.session_state.df is not None:
        count = len(st.session_state.df)
        today = datetime.date.today().isoformat()
        st.markdown(
            f'<div class="dir-info"><b>{count} members</b> · {today}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown('<div class="sb-section">🔍 Quick Name Search</div>', unsafe_allow_html=True)
        name_search = st.text_input(
            "name_search_field", placeholder="Type a name…", label_visibility="collapsed")
        if name_search.strip():
            df_tmp = st.session_state.df
            nc = _col(df_tmp, "name")
            dc = _col(df_tmp, "desig")
            if nc:
                matches = df_tmp[
                    _series(df_tmp, nc).str.contains(
                        re.escape(name_search.strip()), flags=re.IGNORECASE, regex=True)
                ]
                if matches.empty:
                    st.caption("No matches found.")
                else:
                    for _, row in matches.head(6).iterrows():
                        desg = str(row[dc]) if dc else ""
                        st.markdown(
                            f'<div class="name-match">👤 <b>{row[nc]}</b><br>'
                            f'<span style="font-size:0.76rem;color:#6b7280;">{desg}</span></div>',
                            unsafe_allow_html=True,
                        )
    else:
        st.error("❌ Staff file not found.")

    st.markdown("---")
    st.markdown('<div class="sb-section">⚡ Quick Actions</div>', unsafe_allow_html=True)
    if st.button("👨‍💼 Who is HoD?", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "html": "Who is HoD?"})
        st.session_state.chat_history.append({"role": "bot",  "html": route_query("Who is HoD?")})
        st.rerun()
    if st.button("📋 Generate Tomorrow's Schedule", use_container_width=True):
        q_ = "Assign schedule for tomorrow's exam"
        st.session_state.chat_history.append({"role": "user", "html": q_})
        st.session_state.chat_history.append({"role": "bot",  "html": route_query(q_)})
        st.rerun()
    if st.button("👩‍🏫 List All Faculty", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "html": "List all faculty"})
        st.session_state.chat_history.append({"role": "bot",  "html": route_query("List all faculty")})
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="page-title">🤖 AI AGENT — ACSE DEPARTMENT</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-sub">Staff Directory · Invigilation Query · Schedule Generator · Replacement · WhatsApp Notifier</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="page-divider">', unsafe_allow_html=True)

if st.session_state.chat_history:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
        st.markdown(f'<div class="{cls}">{msg["html"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        "<div style='text-align:center;margin-top:80px;'>"
        "<p style='color:#94a3b8;font-size:1rem;'>"
        "💬 Ask anything — HoD, professors, phone numbers, room assignments, invigilation schedule, replacements."
        "</p>"
        "<p style='color:#cbd5e1;font-size:0.85rem;margin-top:8px;'>"
        "Examples: <i>Who is HoD?</i> · <i>Show all professors</i> · "
        "<i>Who is in room 607?</i> · <i>Assign schedule for tomorrow</i> · "
        "<i>Replace Dr. Kumar</i>"
        "</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([9, 1])
    with col1:
        user_input = st.text_input(
            "q", placeholder="Type your question here…", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("↑")

if submitted and user_input.strip():
    raw = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "html": raw})
    reply = route_query(raw)
    st.session_state.chat_history.append({"role": "bot", "html": reply})
    st.rerun()
