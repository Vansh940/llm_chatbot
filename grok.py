import streamlit as st
import uuid
import time
import json
from threading import Thread
from datetime import datetime
import io
import os
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# reportlab — beautiful chat export PDF
from reportlab.lib.pagesizes  import A4
from reportlab.lib.units       import mm
from reportlab.lib             import colors
from reportlab.lib.styles      import ParagraphStyle
from reportlab.lib.enums       import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable


st.set_page_config(page_title="LLM Chatbot", page_icon="🤖", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Groq Client Setup
# ─────────────────────────────────────────────────────────────────────────────
# Set GROQ_API_KEY in Render environment variables (or .streamlit/secrets.toml locally)
# Fix — replace line 41 with this
try:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME   = "llama-3.1-8b-instant"   # fast & free on Groq

@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not set. Add it in Render → Environment Variables.")
        st.stop()
    return Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# UI Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background-color: #0e1117; }
.chat-title { font-size:32px; font-weight:bold; margin-bottom:2px; }
.chat-sub   { color:gray; margin-bottom:0px; }

section[data-testid="stSidebar"] button {
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    padding:6px 10px; border-radius:8px; text-align:left; width:100%;
}
section[data-testid="stSidebar"] button:hover { background-color:#1f2937; }

/* ── PDF filename pill ─────────────────────────────────────────── */
.pdf-filename {
    display      : inline-flex;
    align-items  : center;
    gap          : 7px;
    background   : rgba(99,179,237,0.12);
    border       : 1px solid rgba(99,179,237,0.30);
    border-radius: 8px;
    padding      : 6px 14px;
    font-size    : 13px;
    color        : #93c5fd;
    white-space  : nowrap;
    overflow     : hidden;
    text-overflow: ellipsis;
    max-width    : 100%;
}
.pdf-none {
    font-size : 13px;
    color     : #6b7280;
    font-style: italic;
    line-height: 2.4;
}

/* ── Compact file-uploader ─────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    padding    : 6px 12px !important;
    min-height : 0        !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div:last-child { display:none; }

/* ── Sticky toolbar wrapper ────────────────────────────────────── */
.sticky-toolbar {
    position     : sticky;
    top          : 0;
    z-index      : 999;
    background   : #0e1117;
    padding      : 8px 0 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 6px;
}

/* ── Analytics: stat cards ─────────────────────────────────────── */
.stat-grid {
    display              : grid;
    grid-template-columns: repeat(4, 1fr);
    gap                  : 16px;
    margin-bottom        : 28px;
}
.stat-card {
    background   : linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border       : 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding      : 22px 16px 18px 16px;
    text-align   : center;
}
.stat-number {
    font-size  : 40px;
    font-weight: 700;
    background : linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.stat-label {
    font-size     : 11px;
    color         : #9ca3af;
    margin-top    : 7px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Analytics: section headings ──────────────────────────────── */
.section-heading {
    font-size    : 16px;
    font-weight  : 600;
    color        : #e5e7eb;
    margin-bottom: 14px;
    display      : flex;
    align-items  : center;
    gap          : 8px;
}

/* ── Analytics: emotion bars ───────────────────────────────────── */
.emo-row    { margin-bottom: 14px; }
.emo-header { display:flex; justify-content:space-between; margin-bottom:5px; }
.emo-name   { font-size:13px; color:#d1d5db; font-weight:500; }
.emo-pct    { font-size:13px; color:#9ca3af; }
.emo-bg     { background:rgba(255,255,255,0.07); border-radius:20px; height:10px; overflow:hidden; }
.emo-fill   { height:10px; border-radius:20px; }

/* ── Analytics: topic chips ───────────────────────────────────── */
.topic-chip {
    display      : inline-flex;
    align-items  : center;
    gap          : 6px;
    background   : rgba(99,102,241,0.15);
    border       : 1px solid rgba(99,102,241,0.30);
    border-radius: 20px;
    padding      : 4px 12px;
    font-size    : 12px;
    color        : #a5b4fc;
    margin       : 4px 4px 4px 0;
}
.topic-count {
    background   : rgba(99,102,241,0.30);
    border-radius: 10px;
    padding      : 1px 7px;
    font-size    : 11px;
    font-weight  : 600;
}

/* ── Analytics: chat activity rows ───────────────────────────────*/
.chat-row {
    display        : flex;
    align-items    : center;
    gap            : 10px;
    padding        : 8px 0;
    border-bottom  : 1px solid rgba(255,255,255,0.05);
}
.chat-row-name  { flex:1; font-size:13px; color:#d1d5db; }
.chat-row-count { font-size:12px; color:#6b7280; }
.chat-bar-wrap  { flex:2; }
.chat-bar-bg    { background:rgba(255,255,255,0.07); border-radius:6px; height:6px; overflow:hidden; }
.chat-bar-fill  { background:linear-gradient(90deg,#3b82f6,#8b5cf6); height:6px; border-radius:6px; }

/* ── Analytics: no-data placeholder ──────────────────────────── */
.no-data {
    text-align : center;
    padding    : 60px 20px;
    color      : #6b7280;
    font-size  : 15px;
}
.no-data-icon { font-size:48px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level stream buffers  (survive st.rerun)
# ─────────────────────────────────────────────────────────────────────────────
if "STREAM_BUFFERS" not in st.__dict__:
    st.STREAM_BUFFERS = {}
    st.STREAM_DONE    = {}

# ─────────────────────────────────────────────────────────────────────────────
# Emotion detection  (LLM-powered via Groq)
# ─────────────────────────────────────────────────────────────────────────────
def detect_emotion(text: str) -> str:
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment classifier. "
                        "Read the user message and reply with ONLY one word: "
                        "positive, negative, or neutral. "
                        "No explanation, no punctuation, just one word."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        result = resp.choices[0].message.content.strip().lower()
        if result in ("positive", "negative", "neutral"):
            return result
    except Exception:
        pass
    return "neutral"

# ─────────────────────────────────────────────────────────────────────────────
# Smart chat naming  (Groq)
# ─────────────────────────────────────────────────────────────────────────────
if "PENDING_TITLES" not in st.__dict__:
    st.PENDING_TITLES = {}

def _generate_title_in_background(first_message: str, chat_id: str) -> None:
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=12,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a chat-title generator. "
                        "Reply with ONLY a 4-word title that captures the topic of the user message. "
                        "No punctuation, no quotes, no explanation — just exactly 4 words."
                    ),
                },
                {"role": "user", "content": first_message},
            ],
        )
        raw_title = resp.choices[0].message.content.strip()
        words     = raw_title.replace("\n"," ").replace('"',"").replace("'","").split()
        title     = " ".join(words[:6]) if words else first_message[:40]
    except Exception:
        title = first_message[:40]
    st.PENDING_TITLES[chat_id] = title

# ─────────────────────────────────────────────────────────────────────────────
# Analytics helpers
# ─────────────────────────────────────────────────────────────────────────────
_STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your",
    "his","its","our","their","this","that","these","those","what","which","who",
    "when","where","why","how","all","both","each","few","more","most","other",
    "some","no","nor","not","only","so","than","too","very","just","but","and",
    "or","if","in","on","at","to","for","of","with","about","by","from","up","out",
    "into","any","also","like","get","want","need","make","know","think","tell",
    "ask","use","go","come","take","give","look","see","now","new","hi","hello",
    "hey","please","thanks","thank","yes","ok","okay","sure","right","good","great",
    "nice","well","back","even","still","way","much","many","something","anything",
    "everything","nothing","using","used","can","able","here","there","then",
}

def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))

def _log_message(chat_id: str, role: str, text: str, emotion: str = "neutral") -> None:
    st.session_state.analytics_log.append({
        "chat_id": chat_id,
        "role"   : role,
        "emotion": emotion,
        "tokens" : _estimate_tokens(text),
        "text"   : text,
    })

def _extract_topics(log: list, top_n: int = 10) -> list:
    words = []
    for entry in log:
        if entry["role"] != "user":
            continue
        for w in entry["text"].lower().split():
            w = w.strip(".,!?;:\"'()[]{}—–")
            if w and w not in _STOPWORDS and len(w) > 3 and w.isalpha():
                words.append(w)
    return Counter(words).most_common(top_n)

def _compute_analytics() -> dict:
    log = st.session_state.get("analytics_log", [])
    if not log:
        return None

    user_msgs = [e for e in log if e["role"] == "user"]
    asst_msgs = [e for e in log if e["role"] == "assistant"]
    total_u   = len(user_msgs) or 1

    emotions  = Counter(e["emotion"] for e in user_msgs)
    topics    = _extract_topics(log)

    chat_msg_counts = Counter(e["chat_id"] for e in log)
    chat_rows = []
    for cid, count in chat_msg_counts.most_common():
        name = st.session_state.chat_names.get(cid, "Deleted Chat")
        chat_rows.append({"id": cid, "name": name, "count": count})

    return {
        "total_msgs"   : len(log),
        "user_msgs"    : len(user_msgs),
        "asst_msgs"    : len(asst_msgs),
        "total_tokens" : sum(e["tokens"] for e in log),
        "active_chats" : len(set(e["chat_id"] for e in log)),
        "emotions"     : emotions,
        "total_user"   : total_u,
        "topics"       : topics,
        "chat_rows"    : chat_rows,
    }

def _emo_bar_html(label: str, icon: str, color: str,
                  count: int, total: int) -> str:
    pct    = round(count / total * 100)
    width  = max(pct, 2)
    return f"""
    <div class="emo-row">
      <div class="emo-header">
        <span class="emo-name">{icon} {label}</span>
        <span class="emo-pct">{pct}% &nbsp;·&nbsp; {count} msg{"s" if count!=1 else ""}</span>
      </div>
      <div class="emo-bg">
        <div class="emo-fill" style="width:{width}%;background:{color};"></div>
      </div>
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# PDF Export (reportlab) — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────
_C_BG        = colors.HexColor("#0e1117")
_C_SURFACE   = colors.HexColor("#1a1d27")
_C_BORDER    = colors.HexColor("#2d3748")
_C_USER_BG   = colors.HexColor("#1e3a5f")
_C_USER_TXT  = colors.HexColor("#bfdbfe")
_C_BOT_BG    = colors.HexColor("#1f2937")
_C_BOT_TXT   = colors.HexColor("#e5e7eb")
_C_ACCENT    = colors.HexColor("#3b82f6")
_C_ACCENT2   = colors.HexColor("#8b5cf6")
_C_POS       = colors.HexColor("#34d399")
_C_NEU       = colors.HexColor("#60a5fa")
_C_NEG       = colors.HexColor("#f87171")
_C_MUTED     = colors.HexColor("#6b7280")
_C_WHITE     = colors.HexColor("#f9fafb")
_C_SUBTEXT   = colors.HexColor("#9ca3af")
_C_DIVIDER   = colors.HexColor("#374151")

_EMPTY_STYLE = ParagraphStyle("empty", fontSize=1, leading=1)
_EMPTY_PARA  = Paragraph("", _EMPTY_STYLE)

class _GradientRect(Flowable):
    def __init__(self, width, height, color_left, color_right):
        super().__init__()
        self.width=width; self.height=height
        self.color_left=color_left; self.color_right=color_right
    def wrap(self, aW, aH):
        return self.width, self.height
    def draw(self):
        steps=60; c=self.canv
        for i in range(steps):
            t=i/steps
            r=self.color_left.red  +t*(self.color_right.red  -self.color_left.red)
            g=self.color_left.green+t*(self.color_right.green-self.color_left.green)
            b=self.color_left.blue +t*(self.color_right.blue -self.color_left.blue)
            c.setFillColorRGB(r,g,b)
            x=self.width*i/steps; w=self.width/steps+1
            c.rect(x,0,w,self.height,fill=1,stroke=0)

def _xml_safe(text):
    return (text.replace("&","&amp;").replace("<","&lt;")
                .replace(">","&gt;").replace('"',"&quot;").replace("'","&#39;"))

def _bubble_table(text, role, emotion, page_w_pts):
    is_user  = role == "user"
    inner_w  = page_w_pts - 40*mm
    bubble_w = min(inner_w * 0.78, 135*mm)
    pad_w    = inner_w - bubble_w

    para_style = ParagraphStyle("bubble", fontName="Helvetica", fontSize=9.5,
        leading=14, textColor=_C_USER_TXT if is_user else _C_BOT_TXT, wordWrap="CJK")
    content_para = Paragraph(_xml_safe(text), para_style)

    emo_icon  = {"positive":"(+)","negative":"(-)","neutral":"(~)"}.get(emotion,"")
    role_label = f"You {emo_icon}" if is_user else "Assistant"
    label_style = ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=8,
        textColor=_C_ACCENT if is_user else _C_MUTED, leading=11)
    label_para = Paragraph(role_label, label_style)

    inner_tbl = Table([[label_para],[content_para]], colWidths=[bubble_w])
    inner_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), _C_USER_BG if is_user else _C_BOT_BG),
        ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
    ]))

    spacer_cell = Paragraph("", ParagraphStyle("sp", fontSize=1, leading=1))

    if is_user:
        cols  = [pad_w, bubble_w]
        cells = [[spacer_cell, inner_tbl]]
    else:
        cols  = [bubble_w, pad_w]
        cells = [[inner_tbl, spacer_cell]]

    outer_tbl = Table([cells], colWidths=cols)
    outer_tbl.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("BACKGROUND",(0,0),(-1,-1),_C_BG),
    ]))
    return KeepTogether([outer_tbl])

def _emo_bar_row(label, pct, bg_color, inner_w):
    bar_w = max(int((inner_w - 50*mm) * pct / 100), 2)
    lbl_p = Paragraph(f"{label}  <b>{pct}%</b>",
        ParagraphStyle("el",fontName="Helvetica",fontSize=8,textColor=_C_SUBTEXT,leading=10))
    bar_tbl = Table([[Paragraph("",ParagraphStyle("b",fontSize=1,leading=1))]],
        colWidths=[bar_w], rowHeights=[6])
    bar_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0),bg_color),
        ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    return [lbl_p, bar_tbl]

def build_chat_export_pdf(chat_name, messages, user_name, analytics_log, chat_id):
    buf = io.BytesIO()
    PAGE_W, PAGE_H = A4
    margin  = 18*mm
    inner_w = PAGE_W - 2*margin

    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=14*mm, bottomMargin=14*mm,
        title=chat_name, author=user_name)
    story = []

    story.append(_GradientRect(inner_w, 22*mm, _C_ACCENT, _C_ACCENT2))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("LLM Chatbot  -  Conversation Report",
        ParagraphStyle("t",fontName="Helvetica-Bold",fontSize=18,
            textColor=_C_WHITE,leading=22,alignment=TA_CENTER)))
    story.append(Spacer(1, 3*mm))

    date_str = datetime.now().strftime("%B %d, %Y  %H:%M")
    story.append(Paragraph(
        f"<b>{_xml_safe(chat_name)}</b> &nbsp;·&nbsp; "
        f"Exported by <b>{_xml_safe(user_name)}</b> &nbsp;·&nbsp; {date_str}",
        ParagraphStyle("meta",fontName="Helvetica",fontSize=9,
            textColor=_C_SUBTEXT,alignment=TA_CENTER,leading=13)))
    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=_C_DIVIDER,spaceAfter=5*mm))

    chat_log  = [e for e in analytics_log if e["chat_id"]==chat_id]
    user_msgs = [e for e in chat_log if e["role"]=="user"]
    total_tok = sum(e["tokens"] for e in chat_log)
    emotions  = Counter(e["emotion"] for e in user_msgs)
    tot_u     = len(user_msgs) or 1
    pos_pct   = round(emotions.get("positive",0)/tot_u*100)
    neu_pct   = round(emotions.get("neutral",0)/tot_u*100)
    neg_pct   = 100-pos_pct-neu_pct

    sl = ParagraphStyle("sl",fontName="Helvetica",fontSize=8,
        textColor=_C_SUBTEXT,alignment=TA_CENTER,leading=11)
    sn = ParagraphStyle("sn",fontName="Helvetica-Bold",fontSize=16,
        textColor=_C_ACCENT,alignment=TA_CENTER,leading=20)

    def _sc(num,lbl): return [Paragraph(str(num),sn),Paragraph(lbl,sl)]

    stats_tbl = Table([[
        _sc(len(messages),"Total Messages"),
        _sc(len(user_msgs),"Your Messages"),
        _sc(f"{total_tok:,}","Est. Tokens"),
        _sc(f"{pos_pct}%","Positive Tone"),
        _sc(f"{neu_pct}%","Neutral Tone"),
        _sc(f"{neg_pct}%","Negative Tone"),
    ]], colWidths=[inner_w/6]*6)
    stats_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),_C_SURFACE),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LINEAFTER",(0,0),(4,0),0.4,_C_BORDER),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(stats_tbl)
    story.append(Spacer(1,5*mm))

    story.append(Paragraph("Emotion Breakdown",
        ParagraphStyle("eh",fontName="Helvetica-Bold",fontSize=10,
            textColor=_C_WHITE,leading=13)))
    story.append(Spacer(1,2*mm))
    emo_tbl = Table([
        _emo_bar_row("Positive (+)", pos_pct, _C_POS, inner_w),
        _emo_bar_row("Neutral  (~)", neu_pct, _C_NEU, inner_w),
        _emo_bar_row("Negative (-)", neg_pct, _C_NEG, inner_w),
    ], colWidths=[42*mm, inner_w-42*mm])
    emo_tbl.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(emo_tbl)
    story.append(Spacer(1,5*mm))
    story.append(HRFlowable(width="100%",thickness=0.4,color=_C_DIVIDER,spaceAfter=5*mm))

    story.append(Paragraph("Conversation",
        ParagraphStyle("ch",fontName="Helvetica-Bold",fontSize=12,
            textColor=_C_WHITE,leading=15)))
    story.append(Spacer(1,3*mm))

    log_emotions = {}
    u_idx = 0
    for msg in messages:
        if msg["role"]=="user" and u_idx<len(user_msgs):
            log_emotions[id(msg)] = user_msgs[u_idx].get("emotion","neutral")
            u_idx += 1

    for msg in messages:
        emo = log_emotions.get(id(msg),"neutral")
        story.append(_bubble_table(msg["content"], msg["role"], emo, PAGE_W))
        story.append(Spacer(1,1.5*mm))

    story.append(Spacer(1,6*mm))
    story.append(HRFlowable(width="100%",thickness=0.4,color=_C_DIVIDER,spaceAfter=3*mm))
    story.append(Paragraph(
        f"Generated by Local LLM Chatbot &nbsp;·&nbsp; Model: {MODEL_NAME} &nbsp;·&nbsp; {date_str}",
        ParagraphStyle("ft",fontName="Helvetica-Oblique",fontSize=8,
            textColor=_C_MUTED,alignment=TA_CENTER,leading=11)))

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(_C_BG)
        canvas.rect(0,0,PAGE_W,PAGE_H,fill=1,stroke=0)
        canvas.setFillColor(_C_MUTED)
        canvas.setFont("Helvetica",7)
        canvas.drawCentredString(PAGE_W/2, 8*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# PDF / RAG helpers  (CPU-only — no cupy needed)
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list:
    words = text.split()
    if len(words) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []
    step   = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks

def _build_tfidf(chunks: list):
    use_idf = len(chunks) >= 4
    vec = TfidfVectorizer(
        stop_words  = "english",
        max_features= 20000,
        ngram_range = (1, 2),
        use_idf     = use_idf,
    )
    matrix = vec.fit_transform(chunks)
    return vec, matrix

def _retrieve(query: str, vec, mat, chunks: list,
              top_k: int = 3, max_chars: int = 4000) -> list:
    q_vec = vec.transform([query])
    sims  = cosine_similarity(q_vec, mat).flatten()
    idxs  = sims.argsort()[::-1]

    result = []
    total  = 0
    min_sim = 0.01 if len(chunks) <= 3 else 0.04

    for idx in idxs:
        if sims[idx] < min_sim:
            break
        chunk = chunks[idx]
        if total + len(chunk) > max_chars:
            continue
        result.append(chunk)
        total += len(chunk)
        if len(result) >= top_k:
            break

    if not result and chunks:
        result = [chunks[idxs[0]]]

    return result

def _process_pdf(pdf_file, chat_id: str) -> bool:
    with st.spinner(f"⚙️ Indexing **{pdf_file.name}** …"):
        raw     = pdf_file.read()
        reader  = PdfReader(io.BytesIO(raw))
        n_pages = len(reader.pages)
        text    = ""
        for n, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if t:
                text += f"\n[Page {n+1}]\n{t}"
        if not text.strip():
            st.error("⚠️ Could not extract text — the PDF might be scanned/image-based.")
            return False
        chunks = _chunk_text(text)
        if not chunks:
            st.error("⚠️ No readable text could be extracted from this PDF.")
            return False
        vec, mat = _build_tfidf(chunks)
        st.session_state.pdf_store[chat_id] = {
            "filename"    : pdf_file.name,
            "chunks"      : chunks,
            "vectorizer"  : vec,
            "tfidf_matrix": mat,
            "page_count"  : n_pages,
        }
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Session-state init
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in [
    ("chats",               {}),
    ("chat_names",          {}),
    ("pending_prompt",      None),
    ("generating_chat_ids", set()),
    ("pdf_store",           {}),
    ("analytics_log",       []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "current_chat" not in st.session_state:
    _cid = str(uuid.uuid4())
    st.session_state.current_chat     = _cid
    st.session_state.chats[_cid]      = []
    st.session_state.chat_names[_cid] = "New Chat"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("💬 Conversations")

if st.sidebar.button("➕ New Chat"):
    _cid = str(uuid.uuid4())
    st.session_state.current_chat     = _cid
    st.session_state.chats[_cid]      = []
    st.session_state.chat_names[_cid] = "New Chat"
    st.rerun()

for chat_id in list(st.session_state.chats.keys()):
    c1, c2 = st.sidebar.columns([4, 1])
    with c1:
        title   = st.session_state.chat_names[chat_id]
        is_gen  = chat_id in st.session_state.generating_chat_ids
        pdf_pfx = "📄 " if chat_id in st.session_state.pdf_store else ""
        label   = f"⏳ {pdf_pfx}{title}" if is_gen else f"{pdf_pfx}{title}"
        if st.button(label, key=f"chat_{chat_id}", help=title):
            st.session_state.current_chat = chat_id
            st.rerun()
    with c2:
        if st.button("🗑️", key=f"del_{chat_id}",
                     disabled=chat_id in st.session_state.generating_chat_ids):
            del st.session_state.chats[chat_id]
            del st.session_state.chat_names[chat_id]
            st.session_state.pdf_store.pop(chat_id, None)
            st.STREAM_BUFFERS.pop(chat_id, None)
            st.STREAM_DONE.pop(chat_id, None)
            st.session_state.generating_chat_ids.discard(chat_id)
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.current_chat not in st.session_state.chats:
    _cid = str(uuid.uuid4())
    st.session_state.current_chat     = _cid
    st.session_state.chats[_cid]      = []
    st.session_state.chat_names[_cid] = "New Chat"

current_chat_id     = st.session_state.current_chat
messages            = st.session_state.chats[current_chat_id]
this_chat_streaming = current_chat_id in st.session_state.generating_chat_ids
has_pdf             = current_chat_id in st.session_state.pdf_store

# Flush pending smart titles
for _tid, _title in list(st.PENDING_TITLES.items()):
    if _tid in st.session_state.chat_names:
        st.session_state.chat_names[_tid] = _title
    del st.PENDING_TITLES[_tid]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="chat-title">🤖 Local LLM Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="chat-sub">Powered by Groq · llama-3.1-8b-instant</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pinned PDF toolbar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sticky-toolbar">', unsafe_allow_html=True)

upload_col, info_col = st.columns([3, 5], gap="medium")

with upload_col:
    uploaded_pdf = st.file_uploader(
        "📎 Upload PDF",
        type             = ["pdf"],
        key              = f"pdf_uploader_{current_chat_id}",
        disabled         = this_chat_streaming,
        label_visibility = "collapsed",
        help             = "Upload a PDF to ask questions about it",
    )
    st.caption("📎 Upload a PDF to enable document Q&A")

with info_col:
    if has_pdf:
        _info = st.session_state.pdf_store[current_chat_id]
        pill_col, btn_col = st.columns([8, 1], gap="small")
        with pill_col:
            st.markdown(
                f'<div class="pdf-filename">'
                f'📄 &nbsp;<strong>{_info["filename"]}</strong>'
                f'&nbsp;·&nbsp;{_info["page_count"]} pages'
                f'&nbsp;·&nbsp;{len(_info["chunks"])} chunks'
                f'</div>',
                unsafe_allow_html=True,
            )
        with btn_col:
            st.write("")
            if st.button("✕", key="remove_pdf",
                         help="Remove PDF", disabled=this_chat_streaming):
                del st.session_state.pdf_store[current_chat_id]
                st.rerun()
    else:
        st.markdown(
            '<p class="pdf-none">No PDF attached — upload one to chat with your document</p>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_pdf is not None:
    stored_name = st.session_state.pdf_store.get(current_chat_id, {}).get("filename")
    if stored_name != uploaded_pdf.name:
        ok = _process_pdf(uploaded_pdf, current_chat_id)
        if ok:
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_analytics = st.tabs(["💬 Chat", "📊 Analytics Dashboard"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if messages:
        _, exp_right = st.columns([5, 3], gap="small")
        with exp_right:
            with st.expander("📥 Export Chat as PDF", expanded=False):
                user_name_input = st.text_input(
                    "Your name (shown on report)",
                    value=st.session_state.get("export_user_name", ""),
                    placeholder="e.g. File Name",
                    key="export_name_field",
                )
                if user_name_input:
                    st.session_state["export_user_name"] = user_name_input

                if st.button(
                    "⬇️ Generate & Download PDF",
                    disabled = this_chat_streaming,
                    type     = "primary",
                    key      = "export_pdf_btn",
                    use_container_width=True,
                ):
                    with st.spinner("Building your PDF report…"):
                        try:
                            pdf_bytes = build_chat_export_pdf(
                                chat_name     = st.session_state.chat_names[current_chat_id],
                                messages      = messages,
                                user_name     = user_name_input or "Anonymous",
                                analytics_log = st.session_state.analytics_log,
                                chat_id       = current_chat_id,
                            )
                            st.download_button(
                                label    = "✅ Click to Save PDF",
                                data     = pdf_bytes,
                                file_name= f"chat_report_{current_chat_id[:8]}.pdf",
                                mime     = "application/pdf",
                                key      = f"dl_{current_chat_id}",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"Export failed: {e}")

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if this_chat_streaming:
        partial = st.STREAM_BUFFERS.get(current_chat_id, "")
        done    = st.STREAM_DONE.get(current_chat_id, False)

        with st.chat_message("assistant"):
            st.markdown((partial or "_Thinking…_") + ("" if done else " ▌"))

        if done:
            messages.append({"role": "assistant", "content": partial})
            _log_message(current_chat_id, "assistant", partial, "neutral")
            st.STREAM_BUFFERS.pop(current_chat_id, None)
            st.STREAM_DONE.pop(current_chat_id, None)
            st.session_state.generating_chat_ids.discard(current_chat_id)
            st.rerun()
        else:
            time.sleep(0.1)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:

    stats = _compute_analytics()

    if not stats:
        st.markdown("""
        <div class="no-data">
          <div class="no-data-icon">📊</div>
          <div><strong>No analytics yet</strong></div>
          <div style="margin-top:8px;font-size:13px;">
            Start chatting to see your message stats, emotion breakdown, and topic trends.
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-number">{stats["total_msgs"]}</div>
            <div class="stat-label">💬 Total Messages</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">{stats["user_msgs"]}</div>
            <div class="stat-label">👤 Your Messages</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">{stats["total_tokens"]:,}</div>
            <div class="stat-label">🔤 Est. Tokens Used</div>
          </div>
          <div class="stat-card">
            <div class="stat-number">{stats["active_chats"]}</div>
            <div class="stat-label">🗂️ Active Chats</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        emo_col, topic_col = st.columns([1, 1], gap="large")

        with emo_col:
            st.markdown('<div class="section-heading">😊 Emotion Breakdown</div>',
                        unsafe_allow_html=True)
            em  = stats["emotions"]
            tot = stats["total_user"]
            pos_n = em.get("positive", 0)
            neu_n = em.get("neutral",  0)
            neg_n = em.get("negative", 0)
            st.markdown(
                _emo_bar_html("Positive", "😊", "#34d399", pos_n, tot) +
                _emo_bar_html("Neutral",  "😐", "#60a5fa", neu_n, tot) +
                _emo_bar_html("Negative", "😞", "#f87171", neg_n, tot),
                unsafe_allow_html=True,
            )
            dominant = max(em, key=em.get) if em else "neutral"
            dominant_pct = round(em.get(dominant, 0) / tot * 100)
            dominant_label = {"positive":"positive 😊","neutral":"neutral 😐",
                              "negative":"negative 😞"}[dominant]
            st.caption(
                f"Overall tone is mostly **{dominant_label}** "
                f"({dominant_pct}% of your messages)."
            )

        with topic_col:
            st.markdown('<div class="section-heading">🔥 Most Used Topics</div>',
                        unsafe_allow_html=True)
            if stats["topics"]:
                chips_html = "".join(
                    f'<span class="topic-chip">{word} '
                    f'<span class="topic-count">{count}</span></span>'
                    for word, count in stats["topics"]
                )
                st.markdown(chips_html, unsafe_allow_html=True)
                st.write("")
                df_topics = pd.DataFrame(
                    stats["topics"], columns=["Topic", "Mentions"]
                ).set_index("Topic")
                st.bar_chart(df_topics, height=220, use_container_width=True)
            else:
                st.info("Chat more to see topic trends!")

        st.divider()

        st.markdown('<div class="section-heading">📁 Chat Activity</div>',
                    unsafe_allow_html=True)
        chat_rows  = stats["chat_rows"]
        max_count  = chat_rows[0]["count"] if chat_rows else 1
        rows_html = ""
        for row in chat_rows:
            bar_pct  = round(row["count"] / max_count * 100)
            display_name = (row["name"][:38] + "…") if len(row["name"]) > 40 else row["name"]
            rows_html += f"""
            <div class="chat-row">
              <span class="chat-row-name">💬 {display_name}</span>
              <div class="chat-bar-wrap">
                <div class="chat-bar-bg">
                  <div class="chat-bar-fill" style="width:{bar_pct}%;"></div>
                </div>
              </div>
              <span class="chat-row-count">{row["count"]} msg{"s" if row["count"]!=1 else ""}</span>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)
        st.write("")
        if st.button("🗑️ Clear Analytics Data", type="secondary"):
            st.session_state.analytics_log = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Pending prompt
# ─────────────────────────────────────────────────────────────────────────────
prompt = None
if st.session_state.pending_prompt and not this_chat_streaming:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ─────────────────────────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything…", disabled=this_chat_streaming)

if user_input and not this_chat_streaming:
    st.session_state.pending_prompt = user_input
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Generation  (Groq streaming + optional RAG)
# ─────────────────────────────────────────────────────────────────────────────
if prompt:
    messages.append({"role": "user", "content": prompt})

    with tab_chat:
        with st.chat_message("user"):
            st.markdown(prompt)

    if st.session_state.chat_names[current_chat_id] == "New Chat":
        st.session_state.chat_names[current_chat_id] = "✍️ Naming…"
        Thread(
            target=_generate_title_in_background,
            args=(prompt, current_chat_id),
            daemon=True,
        ).start()

    emotion = detect_emotion(prompt)
    _log_message(current_chat_id, "user", prompt, emotion)

    st.session_state.generating_chat_ids.add(current_chat_id)
    st.STREAM_BUFFERS[current_chat_id] = ""
    st.STREAM_DONE[current_chat_id]    = False

    sys_msg = (
        "User sounds upset. Respond empathetically."  if emotion == "negative"
        else "User sounds happy. Respond warmly."     if emotion == "positive"
        else "Respond naturally."
    )

    if current_chat_id in st.session_state.pdf_store:
        _pdf     = st.session_state.pdf_store[current_chat_id]
        relevant = _retrieve(
            prompt, _pdf["vectorizer"], _pdf["tfidf_matrix"], _pdf["chunks"], top_k=3
        )
        if relevant:
            context_block = "\n\n---\n\n".join(relevant)
            sys_msg += (
                f"\n\nThe user has uploaded a document titled '{_pdf['filename']}'."
                f"\n\nRelevant excerpts for this query:\n\n{context_block}\n\n"
                f"Use these excerpts to give a detailed, accurate answer. "
                f"Elaborate on the content and cite specific details from the document "
                f"wherever relevant. If the answer is not fully covered by the excerpts, "
                f"say so and offer what you can infer."
            )

    emotion_messages = [{"role": "system", "content": sys_msg}] + messages
    target_id = current_chat_id

    def stream_in_background():
        try:
            client = get_groq_client()
            stream = client.chat.completions.create(
                model    = MODEL_NAME,
                messages = emotion_messages,
                stream   = True,
                max_tokens = 1024,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    st.STREAM_BUFFERS[target_id] += delta
        except Exception as exc:
            st.STREAM_BUFFERS[target_id] += f"\n⚠️ Error: {exc}"
        finally:
            st.STREAM_DONE[target_id] = True

    Thread(target=stream_in_background, daemon=True).start()
    time.sleep(0.1)
    st.rerun()
