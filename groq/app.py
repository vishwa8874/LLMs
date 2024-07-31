import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

key = ""

if 'vector' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    st.session_state.loader = WebBaseLoader("https://python.langchain.com/v0.1/docs/modules/chains/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


st.title('ChatGroq')

llm = ChatGroq(groq_api_key=key, model_name='Gemma-7b-It')

prompts = ChatPromptTemplate.from_template(
    ''' 
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
question: {input}
    '''
)


doc_chain = create_stuff_documents_chain(llm, prompts)

retriver = st.session_state.vector.as_retriever()
retrival_chain = create_retrieval_chain(retriver, doc_chain)

input_text = st.text_input('What is your Query?')

if input_text:
    start_time = time.process_time()
    response = retrival_chain.invoke({'input': input_text})
    print('response time: ', time.process_time() - start_time )
    st.write(response['answer'])


    with st.expander('Document Similarity Search'):

        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('--------------------------------')
