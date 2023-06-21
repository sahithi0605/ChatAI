from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
import os
import streamlit as st
import io
import asyncio

api_key = st.sidebar.text_input("API Key", type="password")

async def storeDocEmbeds(file, filename):
    reader = PdfReader(file)
    corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(corpus)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = FAISS.from_texts(chunks, embeddings)
    
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(vectors, f)

async def getDocEmbeds(file, filename):
    if not os.path.isfile(filename + ".pkl"):
        await storeDocEmbeds(file, filename)
    
    with open(filename + ".pkl", "rb") as f:
        vectors = pickle.load(f)
    
    return vectors

async def retrieve_answer(query, qa):
    result = qa({"question": query, "chat_history": []})
    return result["answer"]

def initialize_vectors(uploaded_file):
    uploaded_file.seek(0)
    file = uploaded_file.read()
    vectors = asyncio.run(getDocEmbeds(io.BytesIO(file), uploaded_file.name))
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=vectors.as_retriever(), return_source_documents=True)
    return qa, vectors

def run_chat_interface():
    st.title("PDF Abstractor :")

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    uploaded_file = st.file_uploader("Choose a file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            qa, vectors = initialize_vectors(uploaded_file)
            st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:
        user_input = st.text_input("Query:", placeholder="e.g: Summarize the paper in a few sentences")

        if user_input:
            output = asyncio.run(retrieve_answer(user_input, qa))
            st.markdown(f"**Query**: {user_input}")
            st.markdown(f"**Answer**: {output}")

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = api_key
    run_chat_interface()
