
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings  # Replaces OpenAIEmbeddings
from langchain_groq import ChatGroq                      # Replaces ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("Error: .env file mein GROQ_API_KEY nahi mili!")
    exit()

# STEP 1: INDEXING (Load, Chunking, Embed , Store) ---

VIDEO_URL = "https://youtu.be/J5_-l7WIO_w?si=mAHzIQIBPIDZG3aX" 

print("\n STEP 1: Transcript Loading...\n ")

try:
    loader = YoutubeLoader.from_youtube_url(
        VIDEO_URL,
        add_video_info=False,
        language=["en", "hi"],
    )
    docs = loader.load()
    print("✅ Transcript aa gayi! (Bina purane error ke)...\n")

except Exception as e:
    print(f"❌ Error: {e}")
    exit()


# Split text
print("Splitting text...\n")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
chunks = splitter.split_documents(docs)

# Embed and Store
print("Performing Embedding of Chunks and Creating Vector Store...\n")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# STEP 2: Retrieval And Augmentation:
print("Now, Performing Retrieval Opearation...\n")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 9})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#  Prompt Template
prompt = PromptTemplate(
    template="""Question: {question}
    
    Answer the following question based only on the provided context.
    If the context is insufficient, just say "I don't know".
    
    Context:
    {context}
    """,
    input_variables=['context', 'question']
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  
    temperature=0.2
)

# Output Parser
parser = StrOutputParser()

# STEP 3: BUILD THE RUNNABLE CHAIN (LCEL):

parallel_chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
)

rag_chain = parallel_chain | prompt | llm | parser

# STEP 4: EXECUTION :

query = "Can you summarize the video briefly and precisely? You can use 2 to 3 paragraph and give me output in english."
print(f"Invoking chain with query: '{query}'...\n")
print("Performing augmentation and generating final response:\n")

# Invoke the chain
response = rag_chain.invoke(query)
print("Final Response:\n")
print(response)
