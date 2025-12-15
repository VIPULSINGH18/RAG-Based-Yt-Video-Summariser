
🎬 YouTube RAG Summarizer & Chatbot:
End-to-End RAG Application powering real-time interaction with video content using Llama-3 & LangChain
📖 Project Overview:
This project is a Retrieval-Augmented Generation (RAG) application designed to solve the "content overload" problem. It allows users to input any YouTube video URL and interact with it through a chat interface. By leveraging Vector Search and Large Language Models (LLMs), the system provides accurate summaries, answers specific questions, and extracts key insights from hours of video content in seconds, bypassing the need to watch the entire footage.

🏗️ Technical Architecture:
The application implements a full RAG pipeline designed for latency and accuracy:

Ingestion & Transcription:

Utilizes YoutubeLoader to extract transcripts (supporting English, Hindi, and Hinglish) directly from video metadata.

Bypasses audio-to-text latency by leveraging native YouTube captions.

Text Splitting & Chunking:

Implements RecursiveCharacterTextSplitter with a chunk size of 2000 tokens and 400 token overlap.

Optimized to maintain context across semantic boundaries for comprehensive summarization.

Vector Embeddings & Storage:

Generates high-dimensional vector embeddings using the HuggingFace all-MiniLM-L6-v2 model.

Stores embeddings in a local FAISS (Facebook AI Similarity Search) vector database for efficient dense retrieval.

Retrieval & Generation:

Performs semantic similarity search (k=9) to retrieve the most relevant transcript segments.

Augments the prompt with retrieved context and passes it to Llama-3.3-70b-versatile (via Groq API) for high-fidelity response generation.

🛠️ Tech Stack & Tools
LLM Orchestration: LangChain (LCEL - LangChain Expression Language)

Frontend UI: Streamlit (Custom CSS, Session State Management)

Inference Engine: Groq Cloud (Ultra-low latency inference)

Model: Meta Llama 3.3 70B

Vector Database: FAISS (CPU-optimized)

Embeddings: HuggingFace Transformers

Environment Management: python-dotenv for secure API key handling.

✨ Key Features
Context-Aware Chat: Maintains session history to answer follow-up questions accurately.

Multi-Language Support: Capable of processing and understanding Hindi/Hinglish videos and responding in English.

Optimized Retrieval: Tuned k-value retrieval parameters to balance between context window limits (TPM) and answer quality.

Interactive UI: Features real-time processing indicators, animated waiting states, and debug options to view retrieved context chunks.

🚀 Engineering Competencies Demonstrated
RAG Pipeline Construction: Designing a system that reduces LLM hallucinations by grounding answers in retrieved data.

Vector Database Management: Understanding dense vector spaces and similarity search algorithms.

Prompt Engineering: Designing robust system prompts to enforce constraints ("Answer only from context").

API Optimization: Managing Rate Limits (TPM/RPM) on Groq's cloud infrastructure.

State Management: Handling user sessions and caching resources in Streamlit to minimize redundant computation.

👨‍💻 Author
[Vipul Kumar Singh] Aspiring AI Engineer | Data Scientist

Built with passion to solve real-world information retrieval challenges.
