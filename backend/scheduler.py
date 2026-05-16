import re
import random
import datetime
import pandas as pd
import streamlit as st
from .search import _col, _series, _norm

def generate_schedule(df: pd.DataFrame, raw_query: str, parsed: dict) -> str:
    if df is None or df.empty:
        return "<p style='color:#ef4444;'>❌ Staff data not loaded.</p>"

    date_str = parsed.get("date")
    if date_str == "tomorrow" or "tomorrow" in raw_query.lower():
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    elif date_str:
        target_date = date_str
    else:
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    room_matches = re.findall(r'\b(\d{3,4})\b', raw_query)
    
    # Floor detection
    floor_str = parsed.get("floor")
    if not floor_str:
        # Match '6th floor', '6 floor', 'floor 6', and common typos like 'flour'
        floor_match = re.search(r'(?:floor|flour)\s*(\d+)|(\d+)\s*(st|nd|rd|th)?\s*(?:floor|flour)', raw_query.lower())
        if floor_match:
            floor_str = floor_match.group(1) or floor_match.group(2)

    default_rooms = [
        "301", "302", "303", "401", "402", "403",
        "501", "502", "601", "602", "603", "607"
    ]

    if room_matches:
        rooms = room_matches
    elif floor_str:
        # Filter rooms by floor (e.g. '6' -> rooms starting with '6')
        # We handle string matching to catch '6' matching '601', '602', etc.
        rooms = [r for r in default_rooms if r.startswith(str(floor_str))]
        if not rooms:
            # Fallback if no rooms match the floor in our default list
            rooms = default_rooms
    else:
        rooms = default_rooms

    name_col = _col(df, "name")
    desg_col = _col(df, "desig")
    phone_col = _col(df, "mobile", "phone", "contact")
    
    if not name_col:
        return "<p style='color:#ef4444;'>❌ Name column not found in staff data.</p>"

    exclude_roles = ["attender", "lab technician", "peon", "clerk", "dean", "hod", "jr. assistant", "technician"]
    faculty_df = df.copy()
    if desg_col:
        for role in exclude_roles:
            faculty_df = faculty_df[
                ~_series(faculty_df, desg_col).str.lower().str.contains(role, na=False)
            ]

    faculty_names = _series(faculty_df, name_col).tolist()
    faculty_names = [f for f in faculty_names if f.strip() and f.lower() not in ("nan", "none", "name", "designation")]

    if not faculty_names:
        return "<p style='color:#ef4444;'>❌ No faculty available for scheduling.</p>"

    priority_faculty = []
    q_norm = _norm(raw_query)
    for f in faculty_names:
        f_norm = _norm(f)
        if f_norm and f_norm in q_norm:
            priority_faculty.append(f)
    
    # Remove priority from main list
    other_faculty = [f for f in faculty_names if f not in priority_faculty]

    phone_map = _get_phone_map(df, name_col)
    
    sessions = ["FN (9:00 AM - 12:00 PM)", "AN (2:00 PM - 5:00 PM)"]
    schedule = []
    assigned_today = set()
    
    for session in sessions:
        # For each session, shuffle the non-priority faculty
        random.shuffle(other_faculty)
        # Combine: Priority first, then others. 
        # Filter out people already assigned today to avoid double duty.
        current_pool = [f for f in (priority_faculty + other_faculty) if f not in assigned_today]
        
        # If pool is empty (everyone busy), fallback to all faculty
        if not current_pool:
            current_pool = priority_faculty + other_faculty
        
        for i, room in enumerate(rooms):
            faculty = current_pool[i % len(current_pool)]
            assigned_today.add(faculty)
            phone = phone_map.get(faculty, "")
            msg = (
                f"*ACSE Department - Invigilation Duty*\n\n"
                f"Dear {faculty},\n\n"
                f"You have been assigned invigilation duty:\n"
                f"Date: {target_date}\n"
                f"Room: {room}\n"
                f"Session: {session}\n\n"
                f"Please report to the exam cell 15 mins early.\n"
                f"Thank you."
            )
            import urllib.parse
            encoded_msg = urllib.parse.quote(msg)
            wa_link = f"<a href='https://wa.me/{phone}?text={encoded_msg}' target='_blank' style='text-decoration:none;font-size:1.1rem;' title='Notify via WhatsApp'>📲</a>" if phone else "—"
            
            schedule.append({
                "Date": target_date,
                "Session": session,
                "Room No": room,
                "Faculty Assigned": faculty,
                "Status": "Scheduled",
                "Notify": wa_link
            })

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

    absent_faculty = None
    manual_replacement = None

    # Try to find two names in the query
    # Example: "Replace A with B" or "Replace B in place of A"
    all_faculty_names = [str(r[name_col]).strip() for _, r in df.iterrows()]
    q_norm = _norm(raw_query)
    found_names = []
    for name in all_faculty_names:
        n_norm = _norm(name)
        if n_norm and len(n_norm) > 4 and n_norm in q_norm:
            found_names.append(name)
    
    # Sort by length descending to avoid matching substrings
    found_names = sorted(list(set(found_names)), key=len, reverse=True)

    if len(found_names) >= 1:
        # Check which one is currently in the schedule
        in_sched = [n for n in found_names if schedule_df["Faculty Assigned"].str.contains(re.escape(n), case=False).any()]
        if in_sched:
            absent_faculty = in_sched[0] # The one in the schedule is the one being replaced
            # The other one (if any) is the replacement
            others = [n for n in found_names if n != absent_faculty]
            if others:
                manual_replacement = others[0]
    
    if not absent_faculty:
        # Fallback to regex if no names matched the list
        m = re.search(r'replace\s+([A-Za-z\s\.]+?)(?:\s+with|\s+in\s+place\s+of|\s+due|\s+because|$)',
                      raw_query, re.IGNORECASE)
        if m:
            absent_faculty = m.group(1).strip()
            # Try to see if there is a 'with' or 'in place of' part
            m_with = re.search(r'(?:with|in\s+place\s+of)\s+([A-Za-z\s\.]+)', raw_query, re.IGNORECASE)
            if m_with:
                potential_rep = m_with.group(1).strip()
                if potential_rep.lower() != absent_faculty.lower():
                    manual_replacement = potential_rep

    if not absent_faculty:
        return ("<p style='color:#f59e0b;'>⚠️ Please mention the faculty name to replace. "
                "Example: <i>'Replace Dr. Kumar with available faculty'</i></p>")

    # Match using normalized names in the schedule
    absent_norm = _norm(absent_faculty)
    mask = schedule_df["Faculty Assigned"].apply(_norm).str.contains(
        re.escape(absent_norm), na=False)
    
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

    if manual_replacement:
        replacement = manual_replacement
    else:
        replacement = random.choice(available)
    
    phone_map = _get_phone_map(df, name_col)
    
    # Professional message for replacement
    rep_msg = (
        f"*ACSE Department - Invigilation Replacement*\n\n"
        f"Dear {replacement},\n\n"
        f"You have been assigned as an invigilation replacement for *{absent_faculty}*.\n\n"
        f"Date: {schedule_df[mask]['Date'].iloc[0]}\n"
        f"Room: {', '.join(schedule_df[mask]['Room No'].tolist())}\n"
        f"Session: {', '.join(schedule_df[mask]['Session'].tolist())}\n\n"
        f"Please report to the exam cell.\n"
        f"Thank you."
    )
    import urllib.parse
    encoded_rep_msg = urllib.parse.quote(rep_msg)
    rep_wa_link = f"https://wa.me/{phone_map.get(replacement, '')}?text={encoded_rep_msg}"

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
    ✉️ <a href="{rep_wa_link}" target="_blank" style="color:#166534;font-weight:700;">Click here to send WhatsApp Message</a>
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


def _get_phone_map(df: pd.DataFrame, name_col: str) -> dict:
    phone_col = _col(df, "mobile", "phone", "contact")
    phone_map = {}
    if phone_col:
        for _, row in df.iterrows():
            fname = str(row[name_col]).strip()
            fphone = str(row[phone_col]).strip()
            if fphone and fphone.lower() not in ("nan", "none", ""):
                fphone_clean = re.sub(r'\D', '', fphone)
                if len(fphone_clean) == 10:
                    fphone_clean = "91" + fphone_clean
                phone_map[fname] = fphone_clean
    return phone_map
