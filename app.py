import json
import os
import re
from typing import Dict, List, Tuple
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Empathy Prism · Optimistic Multi-Query AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")

# Inject Custom Creative & Optimistic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --font-heading: 'Outfit', -apple-system, sans-serif;
        --font-body: 'Plus Jakarta Sans', -apple-system, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        
        /* Optimistic Warm Palette */
        --bg-warm: #FAF8F5;
        --bg-card: #FFFFFF;
        --text-headline: #1C1917;
        --text-sub: #57534E;
        --text-muted: #78716C;
        
        --warm-coral: #FF5722;
        --warm-tangerine: #F97316;
        --warm-amber: #F59E0B;
        --warm-honey: #FEF3C7;
        --warm-peach: #FFEDD5;
        --fresh-mint: #10B981;
        --mint-light: #ECFDF5;
        --sky-azure: #0EA5E9;
        --sky-light: #F0F9FF;
        --violet-spark: #6366F1;
        --violet-light: #EEF2FF;
        
        --border-subtle: #E7E5E4;
        --border-warm: #FDE68A;
        --border-card: #F3E8DF;
    }

    /* Overall App Background & Reset */
    .stApp {
        background: 
            radial-gradient(circle at 5% 5%, rgba(254, 240, 138, 0.45) 0%, transparent 35%),
            radial-gradient(circle at 95% 15%, rgba(254, 215, 170, 0.45) 0%, transparent 40%),
            radial-gradient(circle at 50% 95%, rgba(224, 242, 254, 0.45) 0%, transparent 45%),
            #FAF8F5 !important;
        font-family: var(--font-body) !important;
        color: var(--text-headline) !important;
    }

    /* Header & Main Container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-heading) !important;
        letter-spacing: -0.025em !important;
        color: var(--text-headline) !important;
    }

    p, span, div, label {
        font-family: var(--font-body) !important;
    }

    code {
        font-family: var(--font-mono) !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
        background-color: #F5F5F4 !important;
        color: #EA580C !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #FFFBEB 0%, #FFF7ED 50%, #FEF2F2 100%);
        border: 2px solid #FDE68A;
        border-radius: 24px;
        padding: 32px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 30px -8px rgba(245, 158, 11, 0.12), 0 4px 12px -2px rgba(0, 0, 0, 0.02);
    }
    
    .hero-banner::after {
        content: "☀️";
        position: absolute;
        right: 28px;
        top: 20px;
        font-size: 5rem;
        opacity: 0.18;
        transform: rotate(12deg);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: #FFFFFF;
        border: 1.5px solid #FCD34D;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #B45309;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.1);
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.18;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #9A3412 0%, #EA580C 50%, #D97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #44403C;
        max-width: 860px;
        margin-bottom: 18px;
    }

    /* Pipeline Step Pills */
    .pipeline-flow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        font-size: 0.86rem;
        font-weight: 600;
        color: #57534E;
    }
    .flow-node {
        background: #FFFFFF;
        padding: 6px 14px;
        border-radius: 12px;
        border: 1px solid #E7E5E4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .flow-arrow {
        color: #F59E0B;
        font-weight: 800;
    }

    /* Creative Lens Cards */
    .lens-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1.5px solid #E7E5E4;
        box-shadow: 0 4px 16px -2px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .lens-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px -4px rgba(249, 115, 22, 0.15);
        border-color: #FDBA74;
    }
    .lens-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 4px 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .lens-1 { background: #FFEDD5; color: #C2410C; border: 1px solid #FED7AA; }
    .lens-2 { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .lens-3 { background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }

    .lens-query-text {
        font-size: 0.98rem;
        font-weight: 600;
        color: #1C1917;
        line-height: 1.45;
        margin-bottom: 8px;
    }
    .lens-purpose {
        font-size: 0.78rem;
        color: #78716C;
        line-height: 1.35;
    }

    /* Result Columns & Containers */
    .result-column-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        border-radius: 14px;
        margin-bottom: 14px;
        font-weight: 700;
        font-family: var(--font-heading);
    }
    .baseline-header {
        background: #F5F5F4;
        border: 1.5px solid #E7E5E4;
        color: #57534E;
    }
    .multiquery-header {
        background: linear-gradient(135deg, #ECFDF5 0%, #FEF3C7 100%);
        border: 1.5px solid #A7F3D0;
        color: #065F46;
    }

    /* Document Result Cards */
    .doc-result-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
        border: 1.5px solid #E7E5E4;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .doc-result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }
    .doc-mq-boosted {
        border-color: #86EFAC !important;
        background: linear-gradient(180deg, #FFFFFF 0%, #F0FDF4 100%);
    }

    .doc-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        gap: 8px;
    }
    .doc-rank-badge {
        font-size: 0.8rem;
        font-weight: 800;
        padding: 3px 10px;
        border-radius: 9999px;
    }
    .rank-base { background: #E7E5E4; color: #44403C; }
    .rank-gold { background: #FEF08A; color: #854D0E; border: 1px solid #FACC15; }
    .rank-silver { background: #F1F5F9; color: #334155; }
    .rank-bronze { background: #FFEDD5; color: #9A3412; }

    .doc-category-pill {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
        background: #F3F4F6;
        color: #4B5563;
    }

    .doc-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1C1917;
        margin-bottom: 6px;
        line-height: 1.35;
    }

    .doc-excerpt {
        font-size: 0.88rem;
        color: #57534E;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    .score-meter-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #FAF8F5;
        padding: 8px 12px;
        border-radius: 10px;
        border: 1px solid #F5F5F4;
    }
    .score-bar-bg {
        flex: 1;
        height: 8px;
        background: #E7E5E4;
        border-radius: 9999px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 9999px;
    }
    .fill-base {
        background: linear-gradient(90deg, #94A3B8, #64748B);
    }
    .fill-mq {
        background: linear-gradient(90deg, #10B981, #059669);
    }
    .score-val {
        font-size: 0.84rem;
        font-weight: 700;
        font-family: var(--font-mono);
        color: #1C1917;
        min-width: 55px;
        text-align: right;
    }

    .found-by-ribbon {
        margin-top: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: center;
    }
    .found-by-pill {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }

    /* Celebration Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 20px 22px;
        border: 1.5px solid #FDE68A;
        box-shadow: 0 4px 16px -2px rgba(245, 158, 11, 0.08);
        text-align: center;
        position: relative;
    }
    .metric-card-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #78716C;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 2.1rem;
        font-weight: 800;
        font-family: var(--font-heading);
        color: #1C1917;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .metric-lift-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 9999px;
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }

    /* Custom Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F5F5F4;
        padding: 6px;
        border-radius: 16px;
        border: 1px solid #E7E5E4;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 12px;
        padding: 0 20px;
        font-family: var(--font-heading);
        font-weight: 600;
        font-size: 0.95rem;
        color: #78716C;
        border: none !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #EA580C !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06) !important;
    }

    /* Custom Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-family: var(--font-heading) !important;
        font-weight: 600 !important;
        border: 1.5px solid #FDBA74 !important;
        background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%) !important;
        color: #C2410C !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(249, 115, 22, 0.08) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(249, 115, 22, 0.18) !important;
        background: linear-gradient(135deg, #FFEDD5 0%, #FED7AA 100%) !important;
        border-color: #FB923C !important;
    }

    /* Input Fields */
    .stTextInput input {
        border-radius: 14px !important;
        border: 1.5px solid #E7E5E4 !important;
        padding: 12px 16px !important;
        font-family: var(--font-body) !important;
        font-size: 1.02rem !important;
        background: #FFFFFF !important;
        color: #1C1917 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
    }
    .stTextInput input:focus {
        border-color: #F97316 !important;
        box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loading & Vector Store Setup
# ---------------------------------------------------------
@st.cache_data
def load_data() -> Tuple[List[Dict], List[Dict]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("documents", []), data.get("test_dataset", [])


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def setup_chroma_db(_embedder, documents: List[Dict]):
    client = chromadb.Client()
    collection_name = "customer_support_docs"

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [doc["id"] for doc in documents]
    texts = [f"{doc['title']}: {doc['content']}" for doc in documents]
    metadatas = [
        {"id": doc["id"], "title": doc["title"], "category": doc["category"], "content": doc["content"]}
        for doc in documents
    ]

    embeddings = _embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return collection


# ---------------------------------------------------------
# Rule Patterns for Query Expansion
# ---------------------------------------------------------
RULE_PATTERNS = [
    # Auth & Login
    (r"(can'?t|cannot|unable to)\s+(log\s*in|sign\s*in|access)", [
        "account login failure credentials troubleshoot",
        "troubleshoot sign in authentication error",
        "unlock user profile password reset"
    ]),
    (r"(forgot|lost|reset)\s+(password|pass|pin)", [
        "password reset link email verification instructions",
        "change forgotten credentials recovery steps",
        "whitelist support email spam folder reset"
    ]),
    (r"(no|didn'?t\s+get|missing|never\s+arrived)\s+(email|link|message)", [
        "verification email not arriving spam folder whitelist",
        "resend password reset confirmation link",
        "check spam promotions folder notification email"
    ]),
    (r"(locked\s+out|account\s+locked)", [
        "account locked excessive failed login attempts",
        "security lockout timer 30 minutes unlock",
        "contact administrator unlock user profile"
    ]),
    (r"(2fa|mfa|two\s*factor|otp|code)", [
        "two factor authentication verification code invalid",
        "authenticator app clock time sync automatic",
        "emergency backup recovery codes login"
    ]),

    # Billing & Payments
    (r"(deducted|charged)\s+twice|(double|duplicate)\s+(charge|payment)", [
        "request refund for accidental duplicate charges",
        "credit card billed twice transaction reversal",
        "duplicate invoice reimbursement timeline"
    ]),
    (r"(refund|money\s+back|cash\s+back)", [
        "submit refund ticket billing invoice credit",
        "cancel transaction payment reimbursement request",
        "billing dispute duplicate charge reversal"
    ]),
    (r"(declined|payment\s+failed|card\s+error)", [
        "credit card payment declined during checkout",
        "billing zip code address mismatch bank error",
        "update credit card payment method"
    ]),
    (r"(cancel|stop|downgrade)\s+(subscription|plan|renewal|auto\s*renew)", [
        "cancelling subscription auto renewal and plan downgrade",
        "turn off automatic subscription renewal billing",
        "downgrade membership plan account settings"
    ]),

    # App Crashes & Freezing
    (r"(won'?t\s+open|doesn'?t\s+start|fails?\s+to\s+launch)", [
        "application freezes or crashes immediately on startup",
        "open in safe mode holding shift key launch",
        "delete corrupted app cache directory appdata"
    ]),
    (r"(crash|crashing|shuts?\s+down|closes)", [
        "software crash on launch troubleshooting guide",
        "clear local application cache files restart",
        "reinstall desktop application clean directory"
    ]),
    (r"(clear|clean)\s+cache|(glitch|bug|blank\s+screen)", [
        "clear browser and app storage cache cookies",
        "fix blank screen rendering glitches restart client",
        "storage preferences wipe temporary assets"
    ]),
    (r"(slow|lag|freeze|freezes|high\s+cpu|memory\s+leak)", [
        "high CPU and RAM consumption system lag",
        "disable hardware acceleration in preferences",
        "close background tabs improve app performance"
    ]),

    # Audio & Video
    (r"(mic|microphone|mute|can'?t\s+hear|people\s+can'?t\s+hear)", [
        "microphone not working or low input volume voice calls",
        "grant microphone privacy permissions audio settings",
        "check headset hardware mute switch default input"
    ]),
    (r"(camera|webcam|video|black\s+screen)", [
        "webcam not detected black screen in video meetings",
        "camera privacy permissions Windows settings",
        "close background apps using video camera device"
    ]),
    (r"(bluetooth|headset|disconnect)", [
        "bluetooth audio device disconnecting repeatedly",
        "disable power management sleep bluetooth adapter",
        "re-pair wireless headset audio settings"
    ]),

    # Network & Sync
    (r"(sync|offline|conflict)", [
        "cloud file sync conflicts and offline file access",
        "timestamp duplicate conflict version resolution",
        "offline mode local changes sync to cloud"
    ]),
    (r"(upload|timeout|slow\s+connection)", [
        "slow file uploads and connection timeout errors",
        "disable VPN network bandwidth throttling bypass",
        "break large files into smaller upload batches"
    ]),
    (r"(503|server|cannot\s+connect)", [
        "cannot connect to server error 503 service unavailable",
        "check cloud service status page maintenance",
        "flush DNS cache ipconfig flushdns"
    ]),
    (r"(database|db|sql|unable to connect.*database|connect.*database)", [
        "database connection timeout and host network configuration",
        "verify database port firewall credentials and connection string",
        "cannot connect to server backend service unavailable"
    ]),
    (r"password.*(not working|incorrect|invalid|wrong|fails?)", [
        "password reset link email verification instructions",
        "account locked excessive failed login attempts",
        "troubleshoot sign in authentication error"
    ]),
]


def expand_query_rules(query: str) -> List[str]:
    """Generates 3 alternative queries using domain synonym rules and semantic rephrasing."""
    cleaned = query.strip()
    expansions: List[str] = []

    for pattern, candidates in RULE_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            for cand in candidates:
                if cand not in expansions and cand.lower() != cleaned.lower():
                    expansions.append(cand)
                if len(expansions) >= 3:
                    break
        if len(expansions) >= 3:
            break

    words = [w for w in re.findall(r"\w+", cleaned) if len(w) > 2]
    keywords = " ".join(words[:5]) if words else cleaned

    fallbacks = [
        f"how to troubleshoot and resolve {keywords} issues",
        f"{keywords} error symptoms official resolution guide",
        f"step by step fix for {keywords} customer support guide",
    ]

    for fb in fallbacks:
        if len(expansions) < 3 and fb not in expansions:
            expansions.append(fb)

    return expansions[:3]


def expand_query(query: str, api_key: str = "") -> Tuple[List[str], str]:
    if api_key.strip():
        try:
            import urllib.request
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            prompt = (
                f"You are a search query expansion assistant for customer support. "
                f"Given the user question: '{query}', produce exactly 3 alternative search queries "
                f"using technical troubleshooting terms and synonyms. "
                f"Return ONLY the 3 queries separated by newlines with no numbering or bullets."
            )
            payload = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 100
            }).encode("utf-8")

            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                text = result["choices"][0]["message"]["content"].strip()
                lines = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n") if line.strip()]
                if len(lines) >= 3:
                    return lines[:3], "LLM Intelligence"
        except Exception as e:
            st.warning(f"Note: Using built-in rule engine ({e})")

    return expand_query_rules(query), "Optimistic Semantic Engine (Offline)"


# ---------------------------------------------------------
# Search Functions
# ---------------------------------------------------------
def search_baseline(collection, embedder, query: str, top_k: int = 3) -> List[Dict]:
    q_emb = embedder.encode([query], show_progress_bar=False).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)

    results = []
    if res and res["ids"] and len(res["ids"][0]) > 0:
        for idx in range(len(res["ids"][0])):
            doc_id = res["ids"][0][idx]
            dist = res["distances"][0][idx] if "distances" in res and res["distances"] else 0.0
            meta = res["metadatas"][0][idx]
            similarity = max(0.0, 1.0 - dist)
            results.append({
                "id": doc_id,
                "title": meta["title"],
                "category": meta["category"],
                "content": meta["content"],
                "score": similarity,
                "distance": dist,
                "found_by": ["Original Query"]
            })
    return results


def search_multi_query(
    collection,
    embedder,
    original_query: str,
    alternative_queries: List[str],
    top_k: int = 3
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    all_queries = [("Original Query", original_query)] + [
        (f"Lens {i+1}", q) for i, q in enumerate(alternative_queries)
    ]

    per_query_results = {}
    doc_registry = {}
    rrf_scores = {}

    for label, q_text in all_queries:
        q_emb = embedder.encode([q_text], show_progress_bar=False).tolist()
        res = collection.query(query_embeddings=q_emb, n_results=top_k)

        sub_list = []
        if res and res["ids"] and len(res["ids"][0]) > 0:
            for rank, doc_id in enumerate(res["ids"][0], start=1):
                idx = rank - 1
                dist = res["distances"][0][idx] if ("distances" in res and res["distances"] and idx < len(res["distances"][0])) else 0.0
                meta = res["metadatas"][0][idx]
                sim = max(0.0, 1.0 - dist)

                doc_info = {
                    "id": doc_id,
                    "title": meta["title"],
                    "category": meta["category"],
                    "content": meta["content"],
                    "score": sim,
                    "distance": dist
                }
                sub_list.append(doc_info)

                if doc_id not in doc_registry:
                    doc_registry[doc_id] = {
                        "id": doc_id,
                        "title": meta["title"],
                        "category": meta["category"],
                        "content": meta["content"],
                        "best_similarity": sim,
                        "found_by": set()
                    }
                else:
                    doc_registry[doc_id]["best_similarity"] = max(
                        doc_registry[doc_id]["best_similarity"], sim
                    )

                doc_registry[doc_id]["found_by"].add(label)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))

        per_query_results[f"{label}: '{q_text}'"] = sub_list

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

    merged_top_k = []
    for doc_id in sorted_doc_ids[:top_k]:
        info = doc_registry[doc_id]
        merged_top_k.append({
            "id": doc_id,
            "title": info["title"],
            "category": info["category"],
            "content": info["content"],
            "score": info["best_similarity"],
            "rrf_score": rrf_scores[doc_id],
            "found_by": sorted(list(info["found_by"]))
        })

    return merged_top_k, per_query_results


# ---------------------------------------------------------
# Evaluation: Precision@3 and Recall@3
# ---------------------------------------------------------
def run_evaluation(collection, embedder, test_cases: List[Dict]) -> Tuple[List[Dict], Dict]:
    records = []
    baseline_p_sum, baseline_r_sum = 0.0, 0.0
    mq_p_sum, mq_r_sum = 0.0, 0.0

    for test in test_cases:
        query = test["query"]
        relevant_ids = set(test["relevant_ids"])
        k = 3

        baseline_res = search_baseline(collection, embedder, query, top_k=k)
        baseline_retrieved_ids = [d["id"] for d in baseline_res]
        base_hits = relevant_ids.intersection(set(baseline_retrieved_ids))
        base_p = len(base_hits) / k
        base_r = len(base_hits) / len(relevant_ids) if relevant_ids else 0.0

        alt_queries = expand_query_rules(query)
        mq_res, _ = search_multi_query(collection, embedder, query, alt_queries, top_k=k)
        mq_retrieved_ids = [d["id"] for d in mq_res]
        mq_hits = relevant_ids.intersection(set(mq_retrieved_ids))
        mq_p = len(mq_hits) / k
        mq_r = len(mq_hits) / len(relevant_ids) if relevant_ids else 0.0

        baseline_p_sum += base_p
        baseline_r_sum += base_r
        mq_p_sum += mq_p
        mq_r_sum += mq_r

        records.append({
            "Test ID": test["id"],
            "Query": query,
            "Target IDs": ", ".join(relevant_ids),
            "Baseline Retrieved": ", ".join(baseline_retrieved_ids),
            "Baseline P@3": base_p,
            "Baseline R@3": base_r,
            "Multi-Query Retrieved": ", ".join(mq_retrieved_ids),
            "Multi-Query P@3": mq_p,
            "Multi-Query R@3": mq_r,
        })

    num_tests = max(1, len(test_cases))
    summary = {
        "baseline_avg_precision": baseline_p_sum / num_tests,
        "baseline_avg_recall": baseline_r_sum / num_tests,
        "mq_avg_precision": mq_p_sum / num_tests,
        "mq_avg_recall": mq_r_sum / num_tests,
    }
    return records, summary


# ---------------------------------------------------------
# Main UI App
# ---------------------------------------------------------
def main():
    documents, test_cases = load_data()

    with st.spinner("☀️ Warming up the Empathy Vector Space..."):
        embedder = get_embedding_model()
        collection = setup_chroma_db(embedder, documents)

    # 1. Warm Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">
            <span>☀️</span> HUMAN INTENT × CREATIVE MULTI-QUERY RETRIEVAL
        </div>
        <div class="hero-title">The Empathy Prism</div>
        <div class="hero-desc">
            Humans search with frustration, slang, and raw symptoms ("my mic is dead", "money took twice"). 
            Documentation speaks in formal architecture. The <b>Empathy Prism</b> splits a single human phrase into 
            <b>3 creative semantic angles</b> to bridge the vocabulary chasm with warmth, speed, and precision.
        </div>
        <div class="pipeline-flow">
            <div class="flow-node">💬 Colloquial Input</div>
            <span class="flow-arrow">➔</span>
            <div class="flow-node">🌈 3 Creative Lenses</div>
            <span class="flow-arrow">➔</span>
            <div class="flow-node">⚡ ChromaDB Vectors</div>
            <span class="flow-arrow">➔</span>
            <div class="flow-node">🎯 Reciprocal Rank Fusion</div>
            <span class="flow-arrow">➔</span>
            <div class="flow-node" style="border-color:#86EFAC; background:#F0FDF4; color:#166534;">🎉 The Exact Solution</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for Rich Multimodal Exploration
    tab_studio, tab_benchmarks, tab_architecture, tab_vault = st.tabs([
        "✨ Empathy Studio",
        "📊 Benchmark & Metrics",
        "💡 How the Prism Works",
        "📚 Knowledge Base Vault (25 Docs)"
    ])

    # ---------------------------------------------------------
    # TAB 1: EMPATHY STUDIO (Live Interactive Search)
    # ---------------------------------------------------------
    with tab_studio:
        st.markdown("### 🔍 Human Thought Input")

        # Preset Chips for Instant Inspiration
        preset_options = [
            "I cannot log in because my password is not working.",
            "My application is unable to connect to the database.",
            "forgot password and reset email never arrived",
            "money deducted twice need refund immediately",
            "app won't open keeps crashing on launch",
            "people can't hear me voice muted on call",
            "how to stop automatic renewal and cancel plan"
        ]

        # Use Session State to manage selected text
        if "current_query" not in st.session_state:
            st.session_state.current_query = preset_options[0]

        # Quick clickable chips
        st.caption("✨ Or pick a real customer frustration:")
        chip_cols = st.columns(len(preset_options[:5]))
        chips = [
            ("🔐 Password Lost", preset_options[0]),
            ("🔌 DB Connection", preset_options[1]),
            ("✉️ Missing Email", preset_options[2]),
            ("💸 Double Charged", preset_options[3]),
            ("💥 App Crash", preset_options[4]),
        ]
        for col, (label, val) in zip(chip_cols, chips):
            if col.button(label, use_container_width=True):
                st.session_state.current_query = val

        user_query = st.text_input(
            "Type your colloquial question or error symptom:",
            value=st.session_state.current_query,
            key="user_query_input",
            placeholder="e.g. money deducted twice or app won't open..."
        )

        if user_query.strip():
            # 1. Multi-Query Expansion
            expanded_queries, engine_label = expand_query(user_query)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"#### 🌈 The 3 Creative Refractions <span style='font-size:0.85rem; font-weight:500; color:#78716C; margin-left:8px;'>Powered by {engine_label}</span>", unsafe_allow_html=True)

            lens_cols = st.columns(3)
            lens_metadata = [
                ("🔧 Lens 1: Diagnostic Translation", "Translates emotional / symptom-based language into formal technical diagnostics.", "lens-1"),
                ("⚡ Lens 2: Action Catalyst", "Rephrases query around recovery procedures, settings adjustments, and user actions.", "lens-2"),
                ("🌐 Lens 3: Domain Synonym Net", "Casts a wide vocabulary net over alternative system names and related keywords.", "lens-3"),
            ]

            for col, q_text, (title, purpose, tag_class) in zip(lens_cols, expanded_queries, lens_metadata):
                with col:
                    st.markdown(f"""
                    <div class="lens-card">
                        <div>
                            <div class="lens-tag {tag_class}">{title}</div>
                            <div class="lens-query-text">"{q_text}"</div>
                        </div>
                        <div class="lens-purpose">{purpose}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # 2. Vector Searches
            baseline_results = search_baseline(collection, embedder, user_query, top_k=3)
            mq_results, _ = search_multi_query(collection, embedder, user_query, expanded_queries, top_k=3)

            st.markdown("<br><hr style='border:none; border-top:1.5px dashed #FDE68A; margin: 24px 0;'><br>", unsafe_allow_html=True)

            # 3. Side-by-Side Comparison
            col_base, col_mq = st.columns(2)

            with col_base:
                st.markdown("""
                <div class="result-column-header baseline-header">
                    <span>🌧️ The Conventional Retrieval (Single Query)</span>
                    <span style="font-size:0.8rem; font-weight:600; color:#78716C;">Baseline Vector Search</span>
                </div>
                """, unsafe_allow_html=True)

                if not baseline_results:
                    st.warning("No documents retrieved.")
                for idx, doc in enumerate(baseline_results, start=1):
                    rank_class = "rank-gold" if idx == 1 else ("rank-silver" if idx == 2 else "rank-bronze")
                    score_pct = int(doc['score'] * 100)
                    st.markdown(f"""
                    <div class="doc-result-card">
                        <div class="doc-header-row">
                            <span class="doc-rank-badge {rank_class}">Rank #{idx}</span>
                            <span class="doc-category-pill">{doc['category']}</span>
                            <span style="font-family:var(--font-mono); font-size:0.75rem; color:#A8A29E;">{doc['id']}</span>
                        </div>
                        <div class="doc-title">{doc['title']}</div>
                        <div class="doc-excerpt">{doc['content']}</div>
                        <div class="score-meter-wrap">
                            <span style="font-size:0.75rem; font-weight:700; color:#64748B;">SIMILARITY</span>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill fill-base" style="width: {score_pct}%;"></div>
                            </div>
                            <span class="score-val">{doc['score']:.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_mq:
                st.markdown("""
                <div class="result-column-header multiquery-header">
                    <span>☀️ The Optimistic Leap (Multi-Query + RRF)</span>
                    <span style="font-size:0.8rem; font-weight:700; color:#059669;">Rank Fusion & Consensus</span>
                </div>
                """, unsafe_allow_html=True)

                if not mq_results:
                    st.warning("No documents retrieved.")
                for idx, doc in enumerate(mq_results, start=1):
                    rank_class = "rank-gold" if idx == 1 else ("rank-silver" if idx == 2 else "rank-bronze")
                    score_pct = int(doc['score'] * 100)
                    pill_tags = "".join([f"<span class='found-by-pill'>✓ {f}</span>" for f in doc['found_by']])
                    
                    st.markdown(f"""
                    <div class="doc-result-card doc-mq-boosted">
                        <div class="doc-header-row">
                            <span class="doc-rank-badge {rank_class}">Rank #{idx}</span>
                            <span class="doc-category-pill" style="background:#DCFCE7; color:#15803D;">{doc['category']}</span>
                            <span style="font-family:var(--font-mono); font-size:0.75rem; font-weight:700; color:#059669;">RRF: {doc.get('rrf_score', 0):.4f}</span>
                        </div>
                        <div class="doc-title">{doc['title']}</div>
                        <div class="doc-excerpt">{doc['content']}</div>
                        <div class="score-meter-wrap" style="background:#F0FDF4; border-color:#DCFCE7;">
                            <span style="font-size:0.75rem; font-weight:700; color:#059669;">CONFIDENCE</span>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill fill-mq" style="width: {score_pct}%;"></div>
                            </div>
                            <span class="score-val" style="color:#047857;">{doc['score']:.3f}</span>
                        </div>
                        <div class="found-by-ribbon">
                            <span style="font-size:0.74rem; font-weight:700; color:#57534E;">Discovered By:</span>
                            {pill_tags}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: BENCHMARK & METRICS
    # ---------------------------------------------------------
    with tab_benchmarks:
        st.markdown("### 📊 Objective Accuracy & Recall Benchmark")
        st.caption("Evaluated across the 7 manually labeled test cases measuring **Precision@3** and **Recall@3** against ChromaDB vectors.")

        records, summary = run_evaluation(collection, embedder, test_cases)

        base_p = summary['baseline_avg_precision']
        mq_p = summary['mq_avg_precision']
        p_delta = (mq_p - base_p) * 100

        base_r = summary['baseline_avg_recall']
        mq_r = summary['mq_avg_recall']
        r_delta = (mq_r - base_r) * 100

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card" style="border-color:#E2E8F0;">
                <div class="metric-card-title">Baseline Precision@3</div>
                <div class="metric-card-value" style="color:#475569;">{base_p:.1%}</div>
                <div style="font-size:0.75rem; color:#94A3B8;">Single query hit rate</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Multi-Query Precision@3</div>
                <div class="metric-card-value" style="color:#059669;">{mq_p:.1%}</div>
                <span class="metric-lift-badge">▲ {p_delta:+.1f}% Accuracy Lift</span>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-color:#E2E8F0;">
                <div class="metric-card-title">Baseline Recall@3</div>
                <div class="metric-card-value" style="color:#475569;">{base_r:.1%}</div>
                <div style="font-size:0.75rem; color:#94A3B8;">Single query coverage</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Multi-Query Recall@3</div>
                <div class="metric-card-value" style="color:#D97706;">{mq_r:.1%}</div>
                <span class="metric-lift-badge" style="background:#FEF3C7; color:#B45309; border-color:#FDE68A;">▲ {r_delta:+.1f}% Coverage Boost</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>#### 📋 Full Ground-Truth Evaluation Matrix", unsafe_allow_html=True)

        # Render styled table
        for r in records:
            p_lift = (r['Multi-Query P@3'] - r['Baseline P@3'])
            r_lift = (r['Multi-Query R@3'] - r['Baseline R@3'])
            
            p_lift_badge = f"<span class='metric-lift-badge'>+{p_lift:.2f}</span>" if p_lift > 0 else f"<span style='color:#78716C; font-size:0.8rem;'>{r['Multi-Query P@3']:.2f}</span>"
            r_lift_badge = f"<span class='metric-lift-badge'>+{r_lift:.2f}</span>" if r_lift > 0 else f"<span style='color:#78716C; font-size:0.8rem;'>{r['Multi-Query R@3']:.2f}</span>"

            st.markdown(f"""
            <div style="background:#FFFFFF; border-radius:14px; padding:16px 20px; border:1px solid #E7E5E4; margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-weight:700; font-size:1.02rem; color:#1C1917;">[{r['Test ID']}] "{r['Query']}"</span>
                    <span style="font-size:0.8rem; background:#FEF3C7; color:#B45309; padding:2px 8px; border-radius:6px; font-weight:600;">Ground Truth Targets: {r['Target IDs']}</span>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; font-size:0.86rem; margin-top:10px;">
                    <div style="background:#F8FAFC; padding:10px 14px; border-radius:10px; border:1px solid #E2E8F0;">
                        <b>🔴 Baseline:</b> Retrieved <code>{r['Baseline Retrieved']}</code><br>
                        <span style="font-size:0.78rem; color:#64748B;">P@3: {r['Baseline P@3']:.2f} | R@3: {r['Baseline R@3']:.2f}</span>
                    </div>
                    <div style="background:#F0FDF4; padding:10px 14px; border-radius:10px; border:1px solid #BBF7D0;">
                        <b>🟢 Multi-Query:</b> Retrieved <code>{r['Multi-Query Retrieved']}</code><br>
                        <span style="font-size:0.78rem; color:#15803D;">P@3: {p_lift_badge} | R@3: {r_lift_badge}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 3: HOW THE PRISM WORKS
    # ---------------------------------------------------------
    with tab_architecture:
        st.markdown("### 💡 Why Conventional AI Search Breaks Down")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div style="background:#FFFFFF; border-radius:18px; padding:24px; border:1.5px solid #E7E5E4; height:100%;">
                <h4 style="color:#C2410C;">⚠️ The Vocabulary Mismatch Trap</h4>
                <p style="font-size:0.92rem; line-height:1.6; color:#44403C;">
                    When users have a problem, they express pain points and everyday vernacular:
                    <br><br>
                    • <i>"My mic is dead on the meeting"</i><br>
                    • <i>"Money was deducted twice"</i><br>
                    • <i>"Screen turned completely black"</i>
                    <br><br>
                    Standard vector search matches phrases against document vectors. But documentation is written in precise, clinical terminology:
                    <br><br>
                    • <b>"Microphone privacy permissions"</b><br>
                    • <b>"Accidental duplicate charge refund"</b><br>
                    • <b>"Display driver crash & cache corruption"</b>
                    <br><br>
                    Even with modern transformer models like <code>all-MiniLM-L6-v2</code>, the cosine distance between emotional symptom phrases and engineering docs can be too wide, causing relevant docs to sink below the top 3.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div style="background:#FFFFFF; border-radius:18px; padding:24px; border:1.5px solid #86EFAC; height:100%;">
                <h4 style="color:#059669;">☀️ The Optimistic Solution: Refraction & RRF</h4>
                <p style="font-size:0.92rem; line-height:1.6; color:#44403C;">
                    Rather than forcing the human to guess the exact technical keyword, the <b>Empathy Prism</b> instantly refracts the query:
                    <br><br>
                    <b>1. Multi-Angle Refraction:</b> Generates 3 complementary query facets covering diagnostic terms, action steps, and domain synonyms.
                    <br><br>
                    <b>2. Independent ChromaDB Traversal:</b> Dispatches 4 queries (1 original + 3 expansions) in parallel into the 384-dimensional vector space.
                    <br><br>
                    <b>3. Reciprocal Rank Fusion (RRF):</b> Documents that appear in multiple query result lists receive an exponential rank boost:
                </p>
                <div style="background:#F0FDF4; padding:12px 16px; border-radius:10px; font-family:var(--font-mono); font-size:0.85rem; color:#166534; border:1px solid #BBF7D0;">
                    RRF Score(d) = ∑ [ 1 / ( 60 + rank(d, q) ) ]
                </div>
                <p style="font-size:0.86rem; color:#57534E; margin-top:10px;">
                    This consensus ensures high-precision discovery without needing massive or expensive language models.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 4: KNOWLEDGE BASE VAULT
    # ---------------------------------------------------------
    with tab_vault:
        st.markdown("### 📚 Curated Support Knowledge Vault")
        st.caption(f"Currently indexing {len(documents)} high-quality customer support articles in ChromaDB.")

        categories = sorted(list(set(d["category"] for d in documents)))
        selected_cat = st.selectbox("Filter by Category:", ["All Categories"] + categories)

        filtered_docs = documents if selected_cat == "All Categories" else [d for d in documents if d["category"] == selected_cat]

        doc_cols = st.columns(2)
        for i, doc in enumerate(filtered_docs):
            with doc_cols[i % 2]:
                st.markdown(f"""
                <div style="background:#FFFFFF; border-radius:16px; padding:18px 20px; border:1.5px solid #E7E5E4; margin-bottom:14px; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-family:var(--font-mono); font-weight:700; color:#EA580C; font-size:0.82rem;">{doc['id']}</span>
                        <span class="doc-category-pill">{doc['category']}</span>
                    </div>
                    <div style="font-weight:700; font-size:1.02rem; color:#1C1917; margin-bottom:6px;">{doc['title']}</div>
                    <div style="font-size:0.88rem; color:#57534E; line-height:1.5;">{doc['content']}</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
