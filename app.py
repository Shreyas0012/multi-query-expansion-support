import json
import os
import re
from typing import Dict, List, Tuple
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Multi-Query Expansion for Support",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multi-Query Expansion for Customer Support")
st.caption(
    "Overcome customer-bot vocabulary mismatch by generating alternative queries, "
    "retrieving with ChromaDB & `all-MiniLM-L6-v2`, and evaluating Precision@3 & Recall@3."
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")


@st.cache_data
def load_data() -> Tuple[List[Dict], List[Dict]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("documents", []), data.get("test_dataset", [])


@st.cache_resource
def get_embedding_model():
    # Load SentenceTransformer model
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def setup_chroma_db(_embedder, documents: List[Dict]):
    client = chromadb.Client()
    collection_name = "customer_support_docs"

    # Reset collection if re-running
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

    # Compute actual embeddings
    embeddings = _embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return collection


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

    # Check pattern matches
    for pattern, candidates in RULE_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            for cand in candidates:
                if cand not in expansions and cand.lower() != cleaned.lower():
                    expansions.append(cand)
                if len(expansions) >= 3:
                    break
        if len(expansions) >= 3:
            break

    # Fallback template expansions if fewer than 3 matches found
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
    """Expands query using optional OpenAI API if key provided, else deterministic rule engine."""
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
                    return lines[:3], "LLM API (OpenAI)"
        except Exception as e:
            st.warning(f"LLM API call failed ({e}). Falling back to rule-based expansion.")

    return expand_query_rules(query), "Rule-Based Python Engine (No API key needed)"


# ---------------------------------------------------------
# Search Functions
# ---------------------------------------------------------
def search_baseline(collection, embedder, query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve top_k documents using only the original query."""
    q_emb = embedder.encode([query], show_progress_bar=False).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)

    results = []
    if res and res["ids"] and len(res["ids"][0]) > 0:
        for idx in range(len(res["ids"][0])):
            doc_id = res["ids"][0][idx]
            dist = res["distances"][0][idx] if "distances" in res and res["distances"] else 0.0
            meta = res["metadatas"][0][idx]
            # Convert cosine distance to similarity percentage: 1 - distance
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
    """
    Search using original + alternative queries, merge results,
    remove duplicates, and return top_k documents using Reciprocal Rank Fusion (RRF).
    """
    all_queries = [("Original Query", original_query)] + [
        (f"Expanded Query {i+1}", q) for i, q in enumerate(alternative_queries)
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

                # Store metadata
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

                # Reciprocal Rank Fusion: score += 1 / (60 + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))

        per_query_results[f"{label}: '{q_text}'"] = sub_list

    # Sort merged unique documents by RRF score descending
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

        # Baseline search
        baseline_res = search_baseline(collection, embedder, query, top_k=k)
        baseline_retrieved_ids = [d["id"] for d in baseline_res]
        base_hits = relevant_ids.intersection(set(baseline_retrieved_ids))
        base_p = len(base_hits) / k
        base_r = len(base_hits) / len(relevant_ids) if relevant_ids else 0.0

        # Multi-query search
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
            "Customer Query": query,
            "Target Relevant": ", ".join(relevant_ids),
            "Baseline Retrieved": ", ".join(baseline_retrieved_ids),
            "Baseline P@3": f"{base_p:.2f}",
            "Baseline R@3": f"{base_r:.2f}",
            "Multi-Query Retrieved": ", ".join(mq_retrieved_ids),
            "Multi-Query P@3": f"{mq_p:.2f}",
            "Multi-Query R@3": f"{mq_r:.2f}",
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
# Main UI
# ---------------------------------------------------------
def main():
    documents, test_cases = load_data()

    with st.spinner("Indexing knowledge documents in ChromaDB..."):
        embedder = get_embedding_model()
        collection = setup_chroma_db(embedder, documents)

    # Clean preset examples
    sample_options = [
        "I cannot log in because my password is not working.",
        "My application is unable to connect to the database.",
        "forgot password and reset email never arrived",
        "money deducted twice need refund immediately",
        "app won't open keeps crashing on launch",
    ]

    selected_sample = st.selectbox(
        "Select a sample question or type your own below:",
        options=sample_options,
        index=0
    )

    user_query = st.text_input(
        "Customer Support Question:",
        value=selected_sample,
        placeholder="Enter customer support query..."
    )

    if user_query.strip():
        # 1. Multi-Query Expansion
        expanded_queries, _ = expand_query(user_query)

        st.markdown("#### 🔄 Generated Alternative Queries (3)")
        cols_q = st.columns(3)
        for i, (col, q_text) in enumerate(zip(cols_q, expanded_queries), start=1):
            with col:
                st.info(f"**Alt {i}:** {q_text}")

        # 2. Vector Searches in ChromaDB
        baseline_results = search_baseline(collection, embedder, user_query, top_k=3)
        mq_results, _ = search_multi_query(
            collection, embedder, user_query, expanded_queries, top_k=3
        )

        st.markdown("---")

        # 3. Side-by-Side Results
        col_base, col_mq = st.columns(2)

        with col_base:
            st.markdown("### 🔴 Baseline Search (Top 3)")
            st.caption("Single original query vector search in ChromaDB")
            if not baseline_results:
                st.warning("No documents retrieved.")
            for idx, doc in enumerate(baseline_results, start=1):
                with st.container(border=True):
                    st.markdown(f"**#{idx} [{doc['id']}] {doc['title']}**")
                    st.caption(f"Category: `{doc['category']}` | Similarity: **{doc['score']:.4f}**")
                    st.write(doc["content"])

        with col_mq:
            st.markdown("### 🟢 Multi-Query Search (Top 3)")
            st.caption("4 queries searched, deduplicated & ranked via RRF")
            if not mq_results:
                st.warning("No documents retrieved.")
            for idx, doc in enumerate(mq_results, start=1):
                with st.container(border=True):
                    st.markdown(f"**#{idx} [{doc['id']}] {doc['title']}**")
                    matches = ", ".join(doc["found_by"])
                    st.caption(f"Category: `{doc['category']}` | Similarity: **{doc['score']:.4f}** | Matched: `{matches}`")
                    st.write(doc["content"])

        # 4. Evaluation Benchmark (Precision@3 & Recall@3)
        st.markdown("---")
        st.markdown("### 📊 Evaluation Benchmark (Ground-Truth Test Set)")

        _, eval_summary = run_evaluation(collection, embedder, test_cases)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Precision@3", f"{eval_summary['baseline_avg_precision']:.1%}")
        m2.metric(
            "Multi-Query Precision@3",
            f"{eval_summary['mq_avg_precision']:.1%}",
            delta=f"{(eval_summary['mq_avg_precision'] - eval_summary['baseline_avg_precision']) * 100:+.1f}%"
        )
        m3.metric("Baseline Recall@3", f"{eval_summary['baseline_avg_recall']:.1%}")
        m4.metric(
            "Multi-Query Recall@3",
            f"{eval_summary['mq_avg_recall']:.1%}",
            delta=f"{(eval_summary['mq_avg_recall'] - eval_summary['baseline_avg_recall']) * 100:+.1f}%"
        )


if __name__ == "__main__":
    main()
