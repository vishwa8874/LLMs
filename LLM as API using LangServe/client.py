import streamlit as st
import requests

def get_phi3_response(input_text):

    response = requests.post('http://localhost:8080/poem/invoke', 
                             json= {'input': {'topic': input_text}})
    return response.json()['output']

st.title('phi3 LLM as API using Langserve')

input_text = st.text_input('Give a topic for writing a poem')

if input_text:
    st.write(get_phi3_response(input_text))
