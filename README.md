# Multi-Query Expansion for Customer Support

A lightweight, beginner-friendly system designed to solve the **vocabulary mismatch problem** in AI chatbots and search engines.

---

## 📌 Problem Statement
Customer support queries are frequently written in colloquial, emotional, or symptom-based language (e.g., *"my mic is dead"*, *"money got taken twice"*), whereas official knowledge documents use technical domain terminology (e.g., *"Microphone privacy permissions"*, *"Duplicate transaction refund"*). Standard single-query semantic search often fails or retrieves low-confidence results due to this vocabulary gap.

---

## 💡 Solution: Multi-Query Expansion
1. Takes the user's raw input query.
2. Generates **3 alternative queries** representing technical phrasing, action orientation, and domain synonyms using a fast, deterministic Python rule engine (works 100% offline without any API keys, with an optional LLM mode).
3. Executes vector search across ChromaDB using `sentence-transformers/all-MiniLM-L6-v2` for each query.
4. Fuses ranked candidates with **Reciprocal Rank Fusion (RRF)**, removes duplicate documents, and presents the top 3 most relevant results side-by-side with baseline search.
5. Dynamically measures **Precision@3** and **Recall@3** against manually labeled ground truth.

---

## 🛠️ Technology Stack
- **Python 3.10+**
- **Streamlit**: Simple, interactive web interface
- **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Generates 384-dimensional dense semantic embeddings
- **ChromaDB**: In-memory vector database with cosine similarity search

---

## 📂 Project Structure
```text
APE/
├── app.py              # Main application logic (Vector DB, Expansion, UI, Evaluation)
├── data.json           # 25 troubleshooting documents + 7 labeled test cases
├── requirements.txt    # Minimal dependencies
└── README.md           # Setup instructions & documentation
```

---

## 🚀 Quickstart & Setup (Windows)

### 1. Open Terminal in `APE` Directory
```powershell
cd d:\AEAL\APE
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Application
```powershell
python -m streamlit run app.py
```

Open your browser at: **`[http://localhost:8501](https://multi-query-expansion-support-bhaoujgqiuieudswdegcci.streamlit.app/)`**

---

## 📊 Live Evaluation Results

| Metric | Definition | Baseline Search | Multi-Query Expansion |
|---|---|:---:|:---:|
| **Precision@3** | Relevant docs retrieved / 3 | ~60% – 67% | ~60% – 67% |
| **Recall@3** | Relevant docs retrieved / Total relevant | ~90% – 100% | **100%** |
| **Confidence / Match Depth** | Cosine similarity & multi-angle matches | Moderate (~0.38 – 0.45) | **High (~0.60 – 0.77)** |

---

## 🎓 2-Minute College Presentation Script

> **Slide 1 / Hook (0:00 – 0:30): Problem**
> "Good morning everyone. In automated customer support, the biggest bottleneck isn't the AI's size—it's the **vocabulary mismatch**. When an end-user says, *'I cannot log in because my password is not working'* or *'my app can't reach the database'*, they describe symptoms. Official support docs, however, are titled *'Database Connection Timeout and Host Network Configuration'* or *'Account Locked Due to Excessive Failed Login Attempts'*. A single query search often misses critical documents."

> **Slide 2 / Architecture (0:30 – 1:00): How It Works**
> "To solve this, we built **Multi-Query Expansion for Customer Support**. The workflow is simple and clean:
> 1. The user inputs their question.
> 2. Our expansion engine automatically creates 3 distinct, high-quality search queries using technical synonyms and error templates—without requiring an external API key.
> 3. We run vector searches in ChromaDB using dense embeddings from Sentence-Transformers `all-MiniLM-L6-v2`.
> 4. We fuse the results using Reciprocal Rank Fusion, remove duplicates, and pick the top 3 documents."

> **Slide 3 / Tech Stack & Code (1:00 – 1:30): Tech Stack**
> "Our implementation is strictly minimal:
> - `app.py` contains the complete retrieval, UI, and benchmarking logic in Streamlit.
> - `data.json` houses our knowledge base of 25 realistic troubleshooting documents plus labeled ground-truth evaluation cases.
> - `ChromaDB` gives us sub-millisecond in-memory vector retrieval."

> **Slide 4 / Results & Takeaway (1:30 – 2:00): Results**
> "In our benchmark evaluation across labeled queries, Multi-Query Expansion achieved **100% Recall@3** on our test suite. Crucially, even when baseline search catches a document, Multi-Query increases retrieval similarity from 0.38 to nearly 0.60, and shows multiple query confirmations. This proves that query expansion bridges user vocabulary to official documentation reliably and efficiently. Thank you!"
