import json
import datetime

# Attempt to load anthropic. This ensures it doesn't hard-crash if missing, matching app.py logic
try:
    import anthropic
    LLM_OK = True
except Exception:
    LLM_OK = False

def llm_parse_query(query: str, api_key: str) -> dict:
    _default = {"intent": "search", "person": None, "role": None,
                "room": None, "floor": None, "date": None, "slot": None, "action": None}
    if not LLM_OK or not api_key:
        return _default
    
    current_year = datetime.datetime.now().year
    
    system = (
        f"You are an NLP assistant for a college staff directory and invigilation system. Current year is {current_year}. "
        "Given a user query, extract: "
        "intent (search / list / greeting / invigilation / schedule / replace / notify / unknown), "
        "person name if mentioned, "
        "role/designation keyword if mentioned, "
        "room number if mentioned, "
        "floor number if mentioned, "
        "date if mentioned (Convert strictly to YYYY-MM-DD or 'tomorrow' if relative), "
        "slot if mentioned (extract string/digit, e.g. '1', '2', 'SLOT1' -> '1'), "
        "action keyword if mentioned (assign/replace/notify/schedule). "
        'Return ONLY compact JSON: {"intent":"search","person":null,"role":null,'
        '"room":null,"floor":null,"date":null,"slot":null,"action":null}. '
        "Use null for missing fields. No markdown, no explanation."
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp   = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": query}],
        )
        parsed = json.loads(resp.content[0].text.strip())
        for k in ("person", "role", "room", "floor", "date", "slot", "action"):
            if isinstance(parsed.get(k), str) and parsed[k].strip().lower() in ("null", "none", ""):
                parsed[k] = None
        return parsed
    except Exception:
        return _default
