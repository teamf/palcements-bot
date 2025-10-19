import io, re, os, requests, json
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
import dateparser
from datetime import datetime, timedelta
from openai import OpenAI

st.set_page_config(page_title="IIMB Placements Q&A", page_icon="🎓", layout="centered")

# ---------------- CONFIG ----------------
CSV_URL = st.secrets.get("CSV_URL", "")  # set in Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
MODEL = st.secrets.get("MODEL", "gpt-4o-mini")  # fast & cheap

# Column name candidates (tolerant to header edits)
DATE_COL_CANDIDATES = ["Application Deadline", "Deadline", "Last date"]
COMPANY_COL_CANDIDATES = ["Company Name", "Company"]
LINK_COL_CANDIDATES = ["Application Link", "Apply Link", "Link"]
JD_COL_CANDIDATES = ["JDs", "Job Description", "Role / JD"]
TALK_COL_CANDIDATES = ["PPT/Leadership talk", "Talk", "PPT", "Session"]
REMARKS_COL_CANDIDATES = ["Remarks", "Notes", "Comments"]

def pick_col(df, candidates, default=None):
    cols = [str(c).strip() for c in df.columns]
    for name in candidates:
        for col in cols:
            if name.lower() == col.lower():
                return col
    for name in candidates:
        for col in cols:
            if name.lower() in col.lower():
                return col
    return default

@st.cache_data(ttl=60)
def load_df():
    if not CSV_URL:
        return pd.DataFrame(), {}
    r = requests.get(CSV_URL, timeout=10)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("").astype(str).str.strip()

    cols = {
        "company": pick_col(df, COMPANY_COL_CANDIDATES),
        "deadline": pick_col(df, DATE_COL_CANDIDATES),
        "link": pick_col(df, LINK_COL_CANDIDATES),
        "jd": pick_col(df, JD_COL_CANDIDATES),
        "talk": pick_col(df, TALK_COL_CANDIDATES),
        "remarks": pick_col(df, REMARKS_COL_CANDIDATES),
    }

    if cols["deadline"] in df.columns:
        def parse_dt(x):
            if pd.isna(x) or str(x).strip()=="":
                return pd.NaT
            if isinstance(x, (int, float)) and not pd.isna(x):
                try:
                    return pd.to_datetime("1899-12-30") + pd.to_timedelta(int(x), unit="D")
                except:
                    pass
            d = dateparser.parse(str(x))
            return pd.to_datetime(d) if d else pd.NaT
        df["_deadline_parsed"] = df[cols["deadline"]].apply(parse_dt)
    else:
        df["_deadline_parsed"] = pd.NaT

    return df, cols

def normalize(s): return str(s or "").strip().lower()

def intent_from_query(q):
    qn = normalize(q)
    intents = {
        "deadline": bool(re.search(r"\b(deadline|last date|due)\b", qn)),
        "link": bool(re.search(r"\b(link|apply|application)\b", qn)),
        "jd": bool(re.search(r"\b(jd|job description|role details?)\b", qn)),
        "talk": bool(re.search(r"\b(ppt|leadership talk|session|talk)\b", qn)),
        "list_all": bool(re.search(r"(all companies|list all|show all)", qn)),
        "today": "today" in qn,
        "tomorrow": "tomorrow" in qn,
    }
    m = re.search(r"next\s+(\d+)\s*(days?|d)\b", qn)
    window_days = int(m.group(1)) if m else None
    if "this week" in qn: window_days = 7
    if intents["today"]: window_days = 0
    if intents["tomorrow"]: window_days = 1
    intents["window_days"] = window_days
    return intents

def fuzzy_companies(df, company_col, q, topk=10, threshold=60):
    if not company_col: return []
    choices = df[company_col].astype(str).tolist()
    results = process.extract(q, choices, scorer=fuzz.WRatio, limit=topk)
    return [i for (_, score, i) in results if score >= threshold]

def filter_by_company(df, cols, q):
    if not cols.get("company"): return df
    idxs = fuzzy_companies(df, cols["company"], q, topk=12)
    if idxs:
        return df.iloc[idxs]
    tokens = [t for t in re.split(r"\W+", normalize(q)) if len(t) >= 3]
    if not tokens: return df
    hay_cols = [c for c in [cols["company"], cols["jd"], cols["remarks"]] if c]
    mask = df.apply(lambda r: any(any(t in normalize(str(r[c])) for c in hay_cols) for t in tokens), axis=1)
    return df[mask]

def apply_time_filters(df, intents):
    if "_deadline_parsed" not in df.columns: return df
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    sub = df.copy()
    if intents["deadline"] or intents["window_days"] is not None:
        sub = sub[pd.notna(sub["_deadline_parsed"])]
        maxd = intents["window_days"]
        if maxd is None:
            sub = sub[sub["_deadline_parsed"] >= today]
        else:
            sub = sub[(sub["_deadline_parsed"] >= today) &
                      (sub["_deadline_parsed"] <= today + timedelta(days=maxd))]
    return sub.sort_values(by="_deadline_parsed", ascending=True)

def rows_to_json(records, cols):
    out = []
    for _, r in records.iterrows():
        out.append({
            "company": r.get(cols["company"], ""),
            "deadline": (r["_deadline_parsed"].strftime("%a, %d %b %Y") 
                         if pd.notna(r["_deadline_parsed"]) else ""),
            "link": r.get(cols["link"], ""),
            "jd": r.get(cols["jd"], ""),
            "talk": r.get(cols["talk"], ""),
            "remarks": r.get(cols["remarks"], ""),
        })
    return out

def craft_answer_with_gpt(question, context_rows):
    if not OPENAI_API_KEY:
        lines = []
        for r in context_rows[:20]:
            lines.append(f"• {r['company']} — Deadline: {r['deadline'] or '—'} — Link: {r['link'] or '—'}")
        return "Here’s what I found:\n" + "\n".join(lines)

    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "You are an IIM Bangalore placements assistant. "
        "Answer only from the provided JSON rows. If something is missing, say you don't have it. "
        "Be concise. For multiple items, use bullet points."
    )
    user = f"""Question: {question}

Data (JSON rows):
{json.dumps(context_rows, ensure_ascii=False, indent=2)}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# ---------------- UI --------------------
st.title("IIMB Placements Q&A")
st.caption("Ask: “deadline for Bain”, “application link for Axis Bank”, “talks this week”, “show deadlines next 5 days”, “all companies”.")

dfcols = load_df()
if not dfcols or dfcols[0].empty:
    st.error("Sheet not reachable. Check CSV_URL in Secrets.")
else:
    df, cols = dfcols
    with st.expander("Preview data (first 30 rows)"):
        st.dataframe(df.head(30), use_container_width=True)

    q = st.text_input("Your question", placeholder="e.g., deadlines this week for Bain / show application links / JD for McKinsey")
    colA, colB = st.columns([1,1])
    with colA:
        refresh = st.button("Refresh sheet now")
    with colB:
        show_context = st.checkbox("Show matched rows sent to ChatGPT", value=False)

    if refresh:
        load_df.clear()
        st.experimental_rerun()

    if q:
        intents = intent_from_query(q)
        sub = filter_by_company(df, cols, q)

        if intents["deadline"] and len(sub) == len(df) and intents["window_days"] is None:
            intents["window_days"] = 14

        sub = apply_time_filters(sub, intents)

        if intents["list_all"] and not (intents["deadline"] or intents["link"] or intents["jd"] or intents["talk"]):
            sub = df.sort_values(by="_deadline_parsed", ascending=True)

        if sub.empty:
            st.write("I couldn’t find anything for that.\nTry:\n• deadline for Bain\n• application link for Axis\n• show deadlines next 5 days\n• JD for McKinsey\n• talks this week")
        else:
            context_rows = rows_to_json(sub.head(30), cols)
            if show_context:
                st.code(json.dumps(context_rows, indent=2), language="json")
            answer = craft_answer_with_gpt(q, context_rows)
            st.write(answer)
