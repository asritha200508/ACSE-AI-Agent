import re
import datetime
import pandas as pd
import streamlit as st

# We still need spacy and embeddings logic. Since they're loaded in app.py, we can 
# pass them, or attempt to use them from session state if available, but it's cleaner to 
# decouple. Actually, the original search_staff relies on global `nlp` and `embedder`.
# We'll import them locally or rely on session_state/global passing.
# To keep it identical to app.py behavior, we'll try loading here or pass as args.

def _norm(s: str) -> str:
    """Normalize string by lowercasing and removing all non-alphanumeric characters."""
    if not s or not isinstance(s, str): return ""
    # Remove common titles first
    s = re.sub(r'\b(dr|mr|ms|mrs|prof)\.?\b', '', s, flags=re.IGNORECASE)
    return re.sub(r'[^a-z0-9]', '', s.lower())


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


def search_staff(df: pd.DataFrame, parsed: dict, raw_query: str, nlp=None, embedder=None, embeddings=None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    q        = raw_query.lower().strip()
    desg_col = _col(df, "desig")
    name_col = _col(df, "name")

    def _desg_filter(pattern):
        if not desg_col:
            return pd.DataFrame()
        mask = _series(df, desg_col).str.contains(pattern, flags=re.IGNORECASE, regex=True)
        return df[mask]

    # ── Greetings ─────────────────────────────────────────────────────────────
    if parsed.get("intent") == "greeting":
        return pd.DataFrame()

    # ── HoD ───────────────────────────────────────────────────────────────────
    if re.search(r"\bhod\b", q) or "head of department" in q or "head of dept" in q:
        if desg_col:
            col_vals = _series(df, desg_col)
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
        ("teaching assistant",        r"teaching\s*(?:assistant|associate|instructor|assoc)"),
    ]
    # Plain 'professor' must NOT also match associate/assistant professor
    plain_prof_triggers = ("all professor", "show professor", "list professor",
                           "who are professors", "all prof")
    q_has_plain_prof = (
        any(t in q for t in plain_prof_triggers)
        or (
            re.search(r'\bprofessor\b', q)
            and not re.search(r'\b(associate|asst|assistant|senior)\b', q)
        )
    )

    for kw, pattern in tier_map:
        if kw in q:
            hits = _desg_filter(pattern)
            if not hits.empty:
                return hits

    if q_has_plain_prof:
        # Only exact Professor level — exclude Associate/Assistant
        if desg_col:
            col_vals = _series(df, desg_col)
            mask = (
                col_vals.str.contains(r'(?i)\bprofessor\b', regex=True)
                & ~col_vals.str.contains(r'(?i)\bassociate\b', regex=True)
                & ~col_vals.str.contains(r'(?i)\b(?:asst|assistant)\b', regex=True)
            )
            hits = df[mask]
            if not hits.empty:
                return hits

    for kw in ("lecturer", "lab technician", "attender"):
        if kw in q:
            pattern_map = {
                "lecturer": r"\blecturer\b",
                "lab technician": r"lab\s*technician",
                "attender": r"\battender\b",
            }
            hits = _desg_filter(pattern_map[kw])
            if not hits.empty:
                return hits

    # ── All faculty / staff ───────────────────────────────────────────────────
    if any(x in q for x in ("all faculty", "all staff", "list all", "show all", "full list")):
        return df

    # ── Direct name match from raw query (before RAG / spaCy) ───────────────────
    if name_col:
        name_series = _series(df, name_col)
        norm_series = name_series.apply(_norm)
        
        # Strip common prefixes to isolate name tokens
        clean_q = re.sub(r'\b(who is|show|details of|contact of|phone of|'  \
                         r'number of|tell me about|find|get|dr\.?|mr\.?|'   \
                         r'ms\.?|mrs\.?|prof\.?)\b', '', q, flags=re.IGNORECASE).strip()
        
        # Try matching the entire normalized query first
        q_norm = _norm(clean_q)
        if len(q_norm) > 3:
            mask = norm_series.str.contains(re.escape(q_norm), na=False)
            hits = df[mask]
            if not hits.empty:
                return hits

        # Fallback to tokenized matching
        clean_q_spaces = re.sub(r'[^\w\s]', ' ', clean_q).strip()
        tokens = [t.strip() for t in clean_q_spaces.split() if len(t.strip()) > 2]

        for size in range(len(tokens), 1, -1):
            for start in range(len(tokens) - size + 1):
                phrase = "".join(tokens[start:start + size])
                p_norm = _norm(phrase)
                if len(p_norm) > 2:
                    mask = norm_series.str.contains(re.escape(p_norm), na=False)
                    hits = df[mask]
                    if not hits.empty:
                        return hits

    # ── Named person from LLM parsed output ──────────────────────────────────────
    person = parsed.get("person")
    if person and name_col:
        p_norm = _norm(person)
        norm_series = _series(df, name_col).apply(_norm)
        mask = norm_series.str.contains(re.escape(p_norm), na=False)
        if df[mask].shape[0] > 0:
            return df[mask]

    # ── Role from LLM ─────────────────────────────────────────────────────────
    role = parsed.get("role")
    if role and desg_col:
        hits = _desg_filter(re.escape(role))
        if not hits.empty:
            return hits

    # ── RAG semantic search ───────────────────────────────────────────────────
    if embedder is not None and embeddings is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        q_emb = embedder.encode([raw_query], convert_to_numpy=True)
        sims  = cosine_similarity(q_emb, embeddings)[0]
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

    # ── Keyword fallback (least precise — only if nothing else matched) ───────
    words = [w for w in re.split(r"\s+", raw_query) if len(w) > 4]
    # Filter out common query words that are not names
    stopwords = {"staff", "faculty", "department", "working", "members", "total", "count", "number"}
    words = [w for w in words if w not in stopwords]
    
    if words:
        combined = pd.Series(False, index=df.index)
        name_only = df[[name_col]] if name_col else df.select_dtypes(include="object")
        for col in name_only.columns:
            for w in words:
                combined |= _series(df, col).str.contains(
                    re.escape(w), flags=re.IGNORECASE, regex=True)
        hits = df[combined]
        if not hits.empty:
            return hits

    return pd.DataFrame()


def search_invigilation(df_invig: pd.DataFrame, staff_df: pd.DataFrame, raw_query: str, parsed: dict) -> str:
    if df_invig is None or df_invig.empty:
        return "<p style='color:#ef4444;'>❌ Invigilation data not loaded. Check backend/data/Invigilation_Data.xlsx</p>"

    q = raw_query.lower().strip()

    room_col    = _col(df_invig, "room", "report")
    floor_col   = _col(df_invig, "floor")
    name_col    = _col(df_invig, "name", "faculty", "invigilator")
    date_col    = _col(df_invig, "date", "day")
    slot_col    = _col(df_invig, "slot", "session", "time")

    hits = df_invig.copy()
    filters_applied = []

    # 0. Check for "not assigned" / "free" / "no duty" intent
    negative_keywords = ("not", "no duty", "free", "without", "zero", "unassigned", "does not have", "no invigilation", "who has no")
    is_negative = any(kw in q for kw in negative_keywords) or (("not" in q or "no" in q) and ("duty" in q or "assign" in q or "invig" in q))
    
    if is_negative and staff_df is not None:
        sname_col = _col(staff_df, "name")
        if sname_col and name_col:
            # Clean up assigned list: remove empty strings, 'nan', etc.
            assigned_raw = _series(df_invig, name_col).astype(str).str.lower().str.strip().tolist()
            assigned = {a for a in assigned_raw if a and a not in ("nan", "none", "—", "invigilator name")}
            free_staff = []
            for _, srow in staff_df.iterrows():
                fname = str(srow[sname_col]).strip()
                if fname.lower() not in assigned:
                    # Also check for partial matches (sometimes names are slightly different)
                    # This is a bit expensive but safer
                    if not any(fname.lower() in a or a in fname.lower() for a in assigned if len(a) > 5):
                        free_staff.append(fname)
            
            if free_staff:
                html = f"<div class='stats-card'><div class='stats-title'>🆓 Faculty with NO Invigilation Duties ({len(free_staff)})</div>"
                html += "<div class='stats-grid' style='grid-template-columns: 1fr;'>"
                for name in sorted(free_staff):
                    html += f"<div class='stats-item'>👤 {name}</div>"
                html += "</div></div>"
                return html
            else:
                return "<p style='color:#ef4444;'>✅ All faculty members have at least one invigilation duty.</p>"

    # 1. Date filter
    date_str = parsed.get("date")
    if date_str:
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d") if date_str == "tomorrow" else date_str
        if date_col:
            date_series = pd.to_datetime(_series(hits, date_col), errors='coerce').dt.strftime('%Y-%m-%d')
            date_series = date_series.fillna(_series(hits, date_col))
            hits = hits[date_series.str.contains(target_date, na=False)]
            filters_applied.append(f"Date: {target_date}")

    # 2. Room filter
    room_str = parsed.get("room")
    if not room_str:
        # Restrict to exactly 3 digits so years (like 2026) are not matched as rooms
        room_match = re.search(r'\b(\d{3})\b', q)
        if room_match:
            room_str = room_match.group(1)

    if room_str and room_col:
        hits = hits[_series(hits, room_col).str.contains(room_str, na=False)]
        filters_applied.append(f"Room: {room_str}")

    # 3. Floor filter
    floor_str = parsed.get("floor")
    if not floor_str:
        floor_match = re.search(r'(?:floor|flour)\s*(\d+)|(\d+)\s*(st|nd|rd|th)?\s*(?:floor|flour)', q)
        if floor_match:
            floor_str = floor_match.group(1) or floor_match.group(2)
            
    if floor_str and room_col:
        # Strip and derive floor. Match rooms like '101', '119', '101-A'
        # Pattern: starts with floor_str followed by exactly 2 digits
        floor_pattern = f"^{floor_str}\\d{{2}}"
        mask = _series(hits, room_col).str.strip().str.contains(floor_pattern, regex=True, na=False)
        hits = hits[mask]
        filters_applied.append(f"Floor: {floor_str}")

    # 4. Slot filter
    slot_str = parsed.get("slot")
    if slot_str and slot_col:
        hits = hits[_series(hits, slot_col).str.contains(re.escape(slot_str), flags=re.IGNORECASE, na=False)]
        filters_applied.append(f"Slot: {slot_str}")

    # 5. Person filter
    person_str = parsed.get("person")
    if not person_str and name_col and staff_df is not None:
        sname_col = _col(staff_df, "name")
        if sname_col:
            for _, srow in staff_df.iterrows():
                faculty_name = str(srow[sname_col]).strip()
                parts = faculty_name.split()
                if any(len(p) > 3 and p.lower() in q for p in parts):
                    person_str = faculty_name
                    break

    if person_str and name_col:
        p_norm = _norm(person_str)
        norm_series = _series(hits, name_col).apply(_norm)
        mask1 = norm_series.str.contains(re.escape(p_norm), na=False)
        hits_filtered = hits[mask1]
        
        if hits_filtered.empty:
            for part in person_str.split():
                if len(part) > 3:
                    p_part_norm = _norm(part)
                    mask_part = norm_series.str.contains(re.escape(p_part_norm), na=False)
                    hits_filtered = hits[mask_part]
                    if not hits_filtered.empty:
                        break
        hits = hits_filtered
        filters_applied.append(f"Faculty: {person_str}")

    if not filters_applied:
        return _invig_table_html(df_invig, "Full Invigilation Schedule")

    filter_title = " | ".join(filters_applied)
    if hits.empty:
        return f"<p style='color:#ef4444;'>❌ No invigilation found for: <b>{filter_title}</b>.</p>"
    
    return _invig_table_html(hits, f"Invigilation: {filter_title}")


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
