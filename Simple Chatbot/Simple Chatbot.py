from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st
import os

os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = 'True'

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assisant who answers to user queries'),
        ('user', 'question: {question}')
    ]
)

llm = Ollama(model= 'phi3')

output_parser = StrOutputParser()

st.title('Langchain Tutorial')

input_text = st.text_input('What is your query?')

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
