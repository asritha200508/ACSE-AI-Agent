import re
import pandas as pd
import streamlit as st
from .llm import llm_parse_query
from .search import search_staff, search_invigilation, _get
from .scheduler import generate_schedule, replace_faculty

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

        card = f"""
<div class="staff-card">
  <div class="sc-name">👤 {name}</div>
  <div class="sc-row">🏷️ <b>Designation :</b> {desg}</div>
  <div class="sc-row">📞 <b>Phone       :</b> {phone}</div>
</div>"""
        return count_html + card

    # Multiple results → table (only Name, Designation, Phone)
    display_cols = []
    for c in results.columns:
        cl = str(c).lower().strip()
        if any(k in cl for k in ("name", "desig", "phone", "mobile")):
            display_cols.append(c)

    if not display_cols:
        display_cols = [c for c in results.columns if str(c).strip() and "unnamed" not in str(c).lower()][:3]

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


def staff_stats_html(df: pd.DataFrame) -> str:
    """Return a summary card showing staff counts grouped by category."""
    if df is None or df.empty:
        return "<p style='color:#ef4444;'>❌ Staff data not available.</p>"

    desig_col = None
    for c in df.columns:
        if "desig" in c.lower():
            desig_col = c
            break

    total = len(df)

    if desig_col is None:
        return f"<p><b>Total staff: {total}</b></p>"

    # Categorize by designation
    cats = {
        "👨‍🏫 Professors":           0,
        "👩‍🏫 Associate Professors": 0,
        "🧑‍🏫 Assistant Professors": 0,
        "📚 Teaching Staff":        0,
        "🔬 Technical / Lab":       0,
        "🛠️ Support Staff":         0,
        "🧑‍💼 Others":               0,
    }

    for desg in df[desig_col].astype(str):
        d = desg.lower()
        if re.search(r'\bprofessor\b', d) and re.search(r'assoc', d):
            cats["👩‍🏫 Associate Professors"] += 1
        elif re.search(r'\bprofessor\b', d) and re.search(r'asst|assistant', d):
            cats["🧑‍🏫 Assistant Professors"] += 1
        elif re.search(r'\bprofessor\b', d):
            cats["👨‍🏫 Professors"] += 1
        elif re.search(r'asst\.?\s*prof|assistant\s*prof', d):
            cats["🧑‍🏫 Assistant Professors"] += 1
        elif re.search(r'teaching\s*assist|teaching\s*assoc|teaching\s*instruct|prog\.?/', d):
            cats["📚 Teaching Staff"] += 1
        elif re.search(r'lab|technician|cap|era', d):
            cats["🔬 Technical / Lab"] += 1
        elif re.search(r'attender|jr\.?\s*assist|peon|clerk', d):
            cats["🛠️ Support Staff"] += 1
        else:
            cats["🧑‍💼 Others"] += 1

    rows_html = "".join(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:8px 12px;border-radius:8px;margin:4px 0;"
        f"background:#eff6ff;border:1px solid #bfdbfe;'>"
        f"<span style='font-size:0.95rem;color:#1e293b;'>{cat}</span>"
        f"<span style='font-weight:700;color:#1d4ed8;font-size:1.1rem;'>{count}</span>"
        f"</div>"
        for cat, count in cats.items() if count > 0
    )

    return f"""
<div class="staff-card">
  <div class="sc-name">🏢 ACSE Department — Staff Summary</div>
  <div style='font-size:0.85rem;color:#64748b;margin-bottom:12px;'>
    As of current data in the staff list
  </div>
  <div style='background:#1d4ed8;color:#fff;border-radius:10px;padding:12px 16px;
       display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
    <span style='font-size:1rem;font-weight:600;'>📊 Total Staff</span>
    <span style='font-size:2rem;font-weight:800;'>{total}</span>
  </div>
  {rows_html}
</div>"""


def route_query(raw: str, nlp=None, embedder=None, embeddings=None) -> str:

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

    # ── Staff Count / Stats ───────────────────────────────────────────────────
    count_keywords = ("how many", "count", "total staff", "total faculty", "number of",
                      "staff strength", "strength", "statistics", "stats", "summary")
    # Also check for 'how many' + 'working' or 'members'
    is_stats_query = (any(kw in q for kw in count_keywords) or 
                      (all(x in q for x in ("how", "many")) and any(y in q for y in ("working", "members", "staff"))))
    
    if is_stats_query:
        return staff_stats_html(df)

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
                      "timetable", "duty chart", "not assigned", "no duty", "is free")
    if intent == "invigilation" or any(kw in q for kw in invig_keywords):
        return search_invigilation(df_invig, df, raw, parsed)

    # Staff search
    results = search_staff(df, parsed, raw, nlp=nlp, embedder=embedder, embeddings=embeddings)
    return results_to_html(results, is_detail=(is_detail and len(results) == 1))
