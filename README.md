<div align="center">

# 🎬 YouTube RAG Summarizer & Chatbot
### *End-to-End RAG Application powering real-time interaction with video content*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-green?style=for-the-badge&logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Llama 3](https://img.shields.io/badge/Model-Llama_3.3_70B-purple?style=for-the-badge&logo=meta&logoColor=white)
![Groq](https://img.shields.io/badge/Inference-Groq_Cloud-orange?style=for-the-badge&logo=fastapi&logoColor=white)

<br />

<img src="https://via.placeholder.com/800x400.png?text=Add+Your+Screenshot+Here" alt="Project Screenshot" width="800"/>

<br />

Built with passion to solve real-world information retrieval challenges.

[View Demo](#) · [Report Bug](#) · [Request Feature](#)

</div>

---

## 📖 Project Overview

**Stop watching, start chatting.**

This project is a **Retrieval-Augmented Generation (RAG)** application designed to solve the "content overload" problem. Users can input any YouTube video URL and interact with it through a chat interface. 

By leveraging **Vector Search** and **Large Language Models (LLMs)**, the system provides accurate summaries, answers specific questions, and extracts key insights from hours of video content in seconds.

---

## 🏗️ Technical Architecture

The application implements a full **RAG pipeline** optimized for low latency and high accuracy:

### 1. Ingestion & Transcription
* **YoutubeLoader:** Extracts transcripts (supporting English, Hindi, and Hinglish) directly from video metadata.
* **Optimization:** Bypasses traditional audio-to-text latency by leveraging native YouTube captions.

### 2. Text Splitting & Chunking
* **RecursiveCharacterTextSplitter:** Chunks text into **2000 tokens** with a **400 token overlap**.
* **Logic:** Optimized to maintain context across semantic boundaries for comprehensive summarization.

### 3. Vector Embeddings & Storage
* **Model:** Generates high-dimensional vector embeddings using **HuggingFace `all-MiniLM-L6-v2`**.
* **Database:** Stores embeddings in a local **FAISS** (Facebook AI Similarity Search) vector database for efficient dense retrieval.

### 4. Retrieval & Generation
* **Search:** Performs semantic similarity search (`k=9`) to retrieve the most relevant transcript segments.
* **Synthesis:** Augments the prompt with retrieved context and passes it to **Llama-3.3-70b-versatile** (via Groq API) for high-fidelity response generation.

---

## 🛠️ Tech Stack & Tools

| Component | Technology Used |
| :--- | :--- |
| **LLM Orchestration** | LangChain (LCEL - LangChain Expression Language) |
| **Frontend UI** | Streamlit (Custom CSS, Session State Management) |
| **Inference Engine** | Groq Cloud (Ultra-low latency inference) |
| **Model** | Meta Llama 3.3 70B |
| **Vector Database** | FAISS (CPU-optimized) |
| **Embeddings** | HuggingFace Transformers |
| **Environment** | `python-dotenv` for secure API key handling |

---

## ✨ Key Features

* ✅ **Context-Aware Chat:** Maintains session history to answer follow-up questions accurately.
* ✅ **Multi-Language Support:** Processes Hindi/Hinglish videos and generates responses in English.
* ✅ **Optimized Retrieval:** Tuned `k-value` parameters to balance Context Window limits (TPM) vs. Answer Quality.
* ✅ **Interactive UI:** Features real-time processing indicators, animated waiting states, and debug options.

---

## 🚀 Engineering Competencies Demonstrated

This project showcases the following advanced engineering skills:

* **RAG Pipeline Construction:** Designing a system that reduces LLM hallucinations by grounding answers in retrieved data.
* **Vector Database Management:** Understanding dense vector spaces and similarity search algorithms.
* **Prompt Engineering:** Designing robust system prompts to enforce constraints ("Answer only from context").
* **API Optimization:** Managing Rate Limits (TPM/RPM) on Groq's cloud infrastructure.
* **State Management:** Handling user sessions and caching resources in Streamlit to minimize redundant computation.

---

<div align="center">

### 👨‍💻 Author

**Vipul Kumar Singh**

*Aspiring AI Engineer | Data Scientist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/vipulsk04/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/VIPULSINGH18)

</div>
