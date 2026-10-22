import json
import os
import re
from typing import Dict, List, Tuple
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Query Expansion Support",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")

# Clean, modern, highly readable CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean Card Styling */
    .simple-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .query-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        height: 100%;
    }

    .tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
    }
    .tag-blue { background-color: #eff6ff; color: #1d4ed8; }
    .tag-green { background-color: #f0fdf4; color: #15803d; }
    .tag-purple { background-color: #faf5ff; color: #7e22ce; }
    .tag-gray { background-color: #f1f5f9; color: #475569; }

    .doc-title {
        font-weight: 600;
        font-size: 0.98rem;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .doc-content {
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.45;
        margin-bottom: 8px;
    }

    .doc-meta {
        font-size: 0.78rem;
        color: #64748b;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }

    .score-badge {
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        background-color: #f1f5f9;
        color: #0f172a;
    }

    .score-badge-highlight {
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        background-color: #dcfce7;
        color: #166534;
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
# Deterministic Query Expansion Engine
# ---------------------------------------------------------
RULE_PATTERNS = [
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


def expand_query(query: str) -> List[str]:
    """Expands customer query into 3 complementary search variations."""
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
        if len(expansions) >= 3:
            break
        if fb not in expansions and fb.lower() != cleaned.lower():
            expansions.append(fb)

    return expansions[:3]


# ---------------------------------------------------------
# Retrieval Logic
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
            })
    return results


def search_multi_query(
    collection,
    embedder,
    original_query: str,
    alternative_queries: List[str],
    top_k: int = 3
) -> List[Dict]:
    all_queries = [("Original", original_query)] + [
        (f"Alt {i+1}", q) for i, q in enumerate(alternative_queries)
    ]

    doc_registry = {}
    rrf_scores = {}

    for label, q_text in all_queries:
        q_emb = embedder.encode([q_text], show_progress_bar=False).tolist()
        res = collection.query(query_embeddings=q_emb, n_results=top_k)

        if res and res["ids"] and len(res["ids"][0]) > 0:
            for rank, doc_id in enumerate(res["ids"][0], start=1):
                idx = rank - 1
                dist = res["distances"][0][idx] if ("distances" in res and res["distances"] and idx < len(res["distances"][0])) else 0.0
                meta = res["metadatas"][0][idx]
                sim = max(0.0, 1.0 - dist)

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
                # Reciprocal Rank Fusion: 1 / (60 + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))

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

    return merged_top_k


# ---------------------------------------------------------
# Evaluation: Precision@3 and Recall@3
# ---------------------------------------------------------
@st.cache_data
def run_evaluation(_collection, _embedder, test_cases: List[Dict]) -> Tuple[List[Dict], Dict]:
    records = []
    baseline_p_sum, baseline_r_sum = 0.0, 0.0
    mq_p_sum, mq_r_sum = 0.0, 0.0

    for test in test_cases:
        query = test["query"]
        relevant_ids = set(test["relevant_ids"])
        k = 3

        base_res = search_baseline(_collection, _embedder, query, top_k=k)
        base_ids = [d["id"] for d in base_res]
        base_hits = relevant_ids.intersection(set(base_ids))
        base_p = len(base_hits) / k
        base_r = len(base_hits) / len(relevant_ids) if relevant_ids else 0.0

        alt_queries = expand_query(query)
        mq_res = search_multi_query(_collection, _embedder, query, alt_queries, top_k=k)
        mq_ids = [d["id"] for d in mq_res]
        mq_hits = relevant_ids.intersection(set(mq_ids))
        mq_p = len(mq_hits) / k
        mq_r = len(mq_hits) / len(relevant_ids) if relevant_ids else 0.0

        baseline_p_sum += base_p
        baseline_r_sum += base_r
        mq_p_sum += mq_p
        mq_r_sum += mq_r

        records.append({
            "Test ID": test["id"],
            "Customer Query": query,
            "Target Documents": ", ".join(relevant_ids),
            "Baseline Retrieved": ", ".join(base_ids),
            "Baseline P@3": f"{base_p:.2f}",
            "Baseline R@3": f"{base_r:.2f}",
            "Multi-Query Retrieved": ", ".join(mq_ids),
            "Multi-Query P@3": f"{mq_p:.2f}",
            "Multi-Query R@3": f"{mq_r:.2f}",
        })

    num_tests = max(1, len(test_cases))
    summary = {
        "baseline_p": baseline_p_sum / num_tests,
        "baseline_r": baseline_r_sum / num_tests,
        "mq_p": mq_p_sum / num_tests,
        "mq_r": mq_r_sum / num_tests,
    }
    return records, summary


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
def main():
    documents, test_cases = load_data()

    with st.spinner("Loading models and vector index..."):
        embedder = get_embedding_model()
        collection = setup_chroma_db(embedder, documents)

    # Simple, Clear Header
    st.title("🔍 Multi-Query Expansion for Customer Support")
    st.write(
        "Customers describe issues in casual language (e.g. *'money taken twice'* or *'my mic is dead'*), "
        "while support articles use technical titles (e.g. *'Duplicate transaction refund'* or *'Microphone privacy permissions'*). "
        "**Multi-Query Expansion** solves this vocabulary gap by generating 3 alternative technical queries and combining their results."
    )

    st.markdown("---")

    # Step 1: Input Query
    st.subheader("1. Enter or Select a Customer Support Question")

    preset_questions = [
        "I cannot log in because my password is not working.",
        "My application is unable to connect to the database.",
        "money deducted twice need refund immediately",
        "forgot password and reset email never arrived",
        "app won't open keeps crashing on launch",
        "people can't hear me voice muted on call",
        "how to stop automatic renewal and cancel plan",
    ]

    selected_preset = st.selectbox(
        "Choose an example question (or select 'Custom' to write your own):",
        options=["Select from examples..."] + preset_questions,
        index=1
    )

    default_text = "" if selected_preset == "Select from examples..." else selected_preset
    user_query = st.text_input("Customer query:", value=default_text, placeholder="e.g. money deducted twice...")

    if user_query.strip():
        # Step 2: Show Expanded Queries
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("2. Generated Alternative Queries")
        st.caption("The system automatically generates 3 alternative queries using technical keywords, action steps, and synonyms:")

        alt_queries = expand_query(user_query)

        col1, col2, col3 = st.columns(3)
        labels = [
            ("Technical Phrasing", "tag-blue"),
            ("Troubleshooting Action", "tag-green"),
            ("Domain Synonyms", "tag-purple"),
        ]

        for col, q_text, (label_title, tag_class) in zip([col1, col2, col3], alt_queries, labels):
            with col:
                st.markdown(f"""
                <div class="query-box">
                    <span class="tag {tag_class}">{label_title}</span>
                    <div style="font-size:0.92rem; font-weight:500; color:#1e293b; margin-top:4px;">
                        "{q_text}"
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Step 3: Run Searches & Compare Side-by-Side
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("3. Side-by-Side Search Results")

        baseline_results = search_baseline(collection, embedder, user_query, top_k=3)
        mq_results = search_multi_query(collection, embedder, user_query, alt_queries, top_k=3)

        col_base, col_mq = st.columns(2)

        with col_base:
            st.markdown("#### 🔴 Standard Search (1 Query)")
            st.caption("Standard vector search using only the raw user query.")

            if not baseline_results:
                st.info("No documents found.")
            for rank, doc in enumerate(baseline_results, start=1):
                st.markdown(f"""
                <div class="simple-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span class="tag tag-gray">Rank #{rank} · {doc['id']}</span>
                        <span class="tag tag-blue">{doc['category']}</span>
                    </div>
                    <div class="doc-title">{doc['title']}</div>
                    <div class="doc-content">{doc['content']}</div>
                    <div class="doc-meta">
                        Similarity Score: <span class="score-badge">{doc['score']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_mq:
            st.markdown("#### 🟢 Multi-Query Search (4 Queries + RRF)")
            st.caption("Combines original + 3 alternatives with Reciprocal Rank Fusion.")

            if not mq_results:
                st.info("No documents found.")
            for rank, doc in enumerate(mq_results, start=1):
                matched_queries = ", ".join(doc['found_by'])
                st.markdown(f"""
                <div class="simple-card" style="border-color:#bbf7d0; background-color:#f0fdf4;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span class="tag tag-green">Rank #{rank} · {doc['id']}</span>
                        <span class="tag tag-blue">{doc['category']}</span>
                    </div>
                    <div class="doc-title">{doc['title']}</div>
                    <div class="doc-content">{doc['content']}</div>
                    <div class="doc-meta">
                        Similarity: <span class="score-badge-highlight">{doc['score']:.3f}</span>
                        <span>·</span>
                        <span>Matched by: <b>{matched_queries}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Step 4: Overall Test Suite Evaluation
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.subheader("4. Overall Performance Benchmark")
        st.caption("Benchmarked across all 7 ground-truth customer support test cases in the dataset:")

        eval_records, summary = run_evaluation(collection, embedder, test_cases)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Precision@3", f"{summary['baseline_p']:.1%}")
        m2.metric(
            "Multi-Query Precision@3",
            f"{summary['mq_p']:.1%}",
            delta=f"{(summary['mq_p'] - summary['baseline_p']) * 100:+.1f}%"
        )
        m3.metric("Baseline Recall@3", f"{summary['baseline_r']:.1%}")
        m4.metric(
            "Multi-Query Recall@3",
            f"{summary['mq_r']:.1%}",
            delta=f"{(summary['mq_r'] - summary['baseline_r']) * 100:+.1f}% Boost"
        )

        # Collapsible Detailed Tables (Non-Intrusive)
        with st.expander("📋 View Benchmark Evaluation Matrix (All 7 Test Queries)"):
            st.dataframe(eval_records, use_container_width=True)

        with st.expander("📚 Browse All Knowledge Base Documents (25 Articles)"):
            for doc in documents:
                st.markdown(f"**[{doc['id']}] {doc['title']}** (`{doc['category']}`)")
                st.caption(doc["content"])
                st.markdown("---")

        with st.expander("ℹ️ How Does Multi-Query Expansion Work?"):
            st.markdown("""
            1. **Input:** The customer enters a problem described in everyday words (e.g. *"money deducted twice"*).
            2. **Expansion:** A deterministic rule engine generates 3 technical variations covering diagnostic names, action steps, and synonyms (e.g. *"accidental duplicate charges refund"*).
            3. **Vector Search:** Each query searches ChromaDB using dense embeddings (`all-MiniLM-L6-v2`).
            4. **Reciprocal Rank Fusion (RRF):** Documents retrieved by multiple queries receive a reciprocal rank boost, moving the most relevant article to the top.
            """)


if __name__ == "__main__":
    main()
