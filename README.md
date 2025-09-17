# BMW Chatbot with Parts Similarity Finder (Local LLM + Vector Search)

This project consists of a local LLM-powered chatbot interface and a system to find similar automotive parts based on textual descriptions. The chatbot is designed to run efficiently on a local machine (MacBook Air M4) using CPU inference and integrates with a Retrieval-Augmented Generation (RAG) system to answer queries about parts.

---

## 🚀 Project Features

-   💬 Chatbot using LLaMA 3.2B model (via Docker Model Runner)
-   ⚙️ Local inference on CPU (Apple M4 chip)
-   🎯 Retrieval-Augmented Generation (RAG) to fetch accurate part alternatives
-   ✨ Streamlit UI to interact with the chatbot
-   📦 Vector database using ChromaDB
-   🔍 Semantic search using sentence-transformers/all-MiniLM-L6-v2
-   📁 Local Git repository used for version control (not pushed to public platforms)

---

## 🖥️ Machine Specifications

| Component | Details                   |
| --------- | ------------------------- |
| Device    | MacBook Air               |
| Chip      | Apple M4                  |
| RAM       | 24 GB Unified Memory      |
| GPU       | Not applicable            |
| OS        | macOS (latest version)    |
| Inference | CPU-based (Apple Silicon) |

---

## 📊 Task 1 – Chatbot System

-   Built a chatbot using the LLaMA 3.2B model served via Docker.
-   Streamlit interface enables interactive multi-session chat.
-   Git used for professional version control and commits throughout.

---

## 🧠 Task 2 – Parts Similarity Finder

### 1. Data Cleaning:

-   Loaded Parts.csv using delimiter `;`.
-   Removed rows with missing or empty `DESCRIPTION`.
-   Cleaned and de-duplicated descriptions using NLTK.

### 2. Embedding & Vectorization:

-   Used `all-MiniLM-L6-v2` transformer model for sentence embeddings.
-   Saved embeddings in a local ChromaDB vector store.

### 3. Retrieval:

-   Built methods to get top 5 most similar parts by description.
-   Support for retrieving with or without similarity scores.

---

## 🔗 Chatbot + RAG Integration

-   Integrated Chroma-based retrieval in chatbot.
-   Users can ask:  
    “What are similar parts to XYZ?”  
    “Suggest alternatives to ITEM123.”
-   Chatbot fetches and returns top related parts.

---

## ⚠️ Data Processing Challenges

-   CSV format required custom delimiter `;`
-   Some rows had missing or empty descriptions – removed
-   Duplicate descriptions after cleaning – filtered out

---

## 📁 Project Structure

```plaintext
.
├── chroma_db/                          # Local storage for Chroma vector DB
│   ├── 6aca02ed-a43b-472d-9563-4a32d47bd1f0/
│   │   ├── data_level0.bin
│   │   ├── header.bin
│   │   ├── length.bin
│   │   └── link_lists.bin
│   └── chroma.sqlite3                  # SQLite index and metadata
│
├── data/                               # Data files
│   ├── Parts.csv                       # Original raw parts data
│   └── rag_data.csv                    # Cleaned and preprocessed for embeddings
│
├── main.py                             # Streamlit app entry point
│
├── notebooks/                          # Jupyter notebooks for exploration
│   ├── data_analysis.ipynb             # Initial data exploration and cleaning
│   └── local_llm.ipynb                 # Test and exploration with local LLM model
│
├── README.md                           # Project documentation file
│
├── requirements.txt                    # Python dependencies
│
└── src/                                # Source code
    ├── chatbot/                        # Chatbot interface and logic
    │   ├── __init__.py
    │   ├── chatbot.py                  # Initial chatbot logic
    │   └── chatbot_v2.py              # Updated chatbot with RAG functionality
    │
    └── rag/                            # Retrieval & similarity modules
        ├── __init__.py
        └── rag_similarity.py          # Embedding, search, and retrieval logic
```

---

## ▶️ Commands to Run the Project

1. ✅ Install required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2. 📦 Build the Vector Database with all part descriptions:

    ```bash
    python src/rag/rag_similarity.py
    ```

3. 💬 Launch the Streamlit chatbot interface:
    ```bash
    streamlit run main.py
    ```

📝 Note:

-   Step 2 is required only the first time or when you upload or clean new data.
-   Be sure to store the generated vector DB under `chroma_db/` as expected.
