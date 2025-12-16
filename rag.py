from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

#Step 1a - Indexing (Document Ingestion):

video_id = "Gfr50f6ZBvo" 
try:   # in some videos transcripts are blocked so that we can handle this error we are using try except block...
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


# Step 1b - Indexing (Text Splitting):

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

#Step 1c and 1d - Indexing (Embedding Generation and Storing in Vector Store):

embeddings= OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

#Step 2 - Retrieval:

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever.invoke('What is deepmind')

#Step 3 - Augmentation:
prompt= PromptTemplate(
    template= """Question: {query}, answer the following question from provided context.
                 If context is insuffiecient just say dont know...
                 Context: {context}""",
    input_variables=['context','query']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text

final_prompt = prompt.invoke({ "query": question,"context": context_text})

#Step 4 - Generation:
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
result= llm.invoke(final_prompt)
print(result.content)

# now we are going to connect all the component of RAG using chain for proper and scalable workflow....
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
main_chain.invoke('Can you summarize the video')

