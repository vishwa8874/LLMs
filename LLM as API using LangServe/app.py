from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import uvicorn


app = FastAPI(
    title='LangChain Server',
    description='A simple Api server for Phi3 model',
    version='1.0'
)

prompt = ChatPromptTemplate.from_template('Write a poem on {topic} for 50 words.')

llm = Ollama(model='phi3')

add_routes(
    app,
    prompt|llm,
    path='/poem'
)


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8080)