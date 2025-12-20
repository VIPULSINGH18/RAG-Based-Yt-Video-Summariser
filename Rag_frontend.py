import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- NEW IMPORTS FOR COOKIE FIX ---
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube RAG Summarizer",
    page_icon="🎬",
    layout="wide"
)

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- CUSTOM CSS FOR UI & ANIMATION ---
st.markdown("""
    <style>
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        height: 3em;
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: 800;
        font-family: 'Helvetica Neue', sans-serif;
        margin-bottom: 10px;
    }
    
    /* Sub-Header Styling */
    .sub-header {
        font-size: 1.2rem;
        color: #FFFFFF;
        text-align: center;
        font-family: 'Verdana', sans-serif;
        font-weight: 400;
        margin-bottom: 40px;
        opacity: 0.8;
    }

    /* ✨ ANIMATION KEYFRAMES (Floating Effect) */
    @keyframes float {
        0% { transform: translateY(0px); opacity: 0.8; }
        50% { transform: translateY(-5px); opacity: 1; text-shadow: 0px 0px 10px #FF4B4B; }
        100% { transform: translateY(0px); opacity: 0.8; }
    }

    /* Apply Animation to the Waiting Text */
    .floating-text {
        animation: float 2s ease-in-out infinite;
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-top: 20px;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px dashed #444;
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        color: gray;
        font-size: 0.8rem;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: API KEY SETUP ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # PRIVACY FIX: Empty by default
    user_api_key = st.text_input(
        "Groq API Key", 
        type="password",
        placeholder="Paste key here (or leave blank to use .env)",
        help="Get your free key at console.groq.com"
    )
    
    if user_api_key:
        groq_api_key = user_api_key
    else:
        groq_api_key = os.getenv("GROQ_API_KEY")
    
    if groq_api_key:
        st.success("✅ API Key loaded safely")
    else:
        st.warning("⚠️ No API Key found")

    st.markdown("---")
    st.markdown("### 🤖 Model Details")
    st.info("Using: **Llama-3.3-70b-versatile**")
    st.info("Embedding: **all-MiniLM-L6-v2**")
    st.markdown("---")
    st.markdown("### 💡 How to use")
    st.markdown("1. Paste a YouTube URL.")
    st.markdown("2. Click **'Process Video'**.")
    st.markdown("3. Ask any question!")

# --- FUNCTIONS (CACHED) ---

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- ✅ REPLACED FUNCTION: Custom Loader using Cookies ---
def load_video_manually(video_url):
    try:
        # 1. Extract Video ID from URL
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be" in video_url:
            video_id = video_url.split("/")[-1]
        else:
            return None, "Invalid YouTube URL format."
        
        # 2. Fetch Transcript using Cookies (Bypasses IP Block)
        # Make sure 'cookies.txt' is in your repo root!
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, cookies="cookies.txt")
        except Exception as e:
            return None, f"Transcript Error: {str(e)} (Ensure cookies.txt is valid)"

        # 3. Combine text into a Document
        full_text = " ".join([i['text'] for i in transcript])
        docs = [Document(page_content=full_text, metadata={"source": video_url})]
        
        return docs, None
    except Exception as e:
        return None, str(e)

def create_vector_db(video_url):
    # ✅ Using the new manual loader instead of YoutubeLoader
    docs, error_msg = load_video_manually(video_url)
    
    if not docs:
        return None, error_msg

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = splitter.split_documents(docs)

        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store, None
    
    except Exception as e:
        return None, str(e)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- MAIN UI ---

st.markdown('<div class="main-header">🎬 YouTube Video Summarizer & Chat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chat with any YouTube video using RAG & Llama 3</div>', unsafe_allow_html=True)

# Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""

# --- INPUT SECTION (Aligned) ---
col1, col2 = st.columns([3, 1], vertical_alignment="bottom")

with col1:
    video_url = st.text_input("🔗 Paste YouTube Video URL here:", placeholder="https://www.youtube.com/watch?v=...")

with col2:
    process_btn = st.button("▶️ Process Video")

# --- PROCESSING LOGIC ---
if process_btn and video_url:
    if not groq_api_key:
        st.error("❌ Please provide a Groq API Key.")
    else:
        with st.spinner("⏳ Fetching transcript..."):
            if video_url != st.session_state.processed_url:
                vector_store, error_msg = create_vector_db(video_url)
                
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_url = video_url
                    st.success("✅ Video processed! Ask away.")
                else:
                    st.error(f"❌ Error: {error_msg}")
            else:
                st.info("✅ Already processed.")

# --- CHAT SECTION ---
if st.session_state.vector_store is not None:
    st.markdown("---")
    st.subheader("💬 Ask a question")
    
    query = st.text_area("Question:", placeholder="e.g., Summarize the video...", height=100, label_visibility="collapsed")
    ask_btn = st.button("🚀 Get Answer")

    if ask_btn and query:
        with st.spinner("🤖 Thinking..."):
            try:
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 9}
                )

                prompt = PromptTemplate(
                    template="""Question: {question}
                    Answer based ONLY on context. If unknown, say so.
                    Context: {context}""",
                    input_variables=['context', 'question']
                )

                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model="llama-3.3-70b-versatile",
                    temperature=0.2
                )

                chain = (
                    RunnableParallel({
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough()
                    })
                    | prompt | llm | StrOutputParser()
                )

                response = chain.invoke(query)
                st.markdown("### 📝 AI Response:")
                st.success(response)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- WAITING STATE (RESTORED WITH ANIMATION) ---
else:
    # Only show if no video is entered yet
    if not video_url:
        st.markdown('<div class="floating-text">👈 Waiting for video input...</div>', unsafe_allow_html=True)

# --- FOOTER (RESTORED) ---
st.markdown("---")
st.markdown('<div class="footer">Built with 🧠 LangChain, 🦙 Llama 3 & 🚀 Streamlit</div>', unsafe_allow_html=True)
