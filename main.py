__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import streamlit as st 
import pandas as pd 
from io import StringIO
from langchain import hub
import tempfile
import os 

# 제목 
st.title('ChatPDF')
st.write('-----')

# OpenAI 키 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type='password')

# 파일 업로드 
uploaded_file = st.file_uploader('PDF 파일을 올려주세요!', type=['pdf'])
st.write('-----')

# #Loader
# loader = PyPDFLoader('unsu.pdf')
# pages = loader.load_and_split() 

def pdf_to_document(upload_file):
  temp_dir = tempfile.TemporaryDirectory() 
  temp_filepath = os.path.join(temp_dir.name, upload_file.name)
  with open(temp_filepath, "wb") as f:
    f.write(upload_file.getvalue())
  loader = PyPDFLoader(temp_filepath)
  pages = loader.load_and_split()
  return pages  

# 업로드된 파일 처리 
if uploaded_file is not None: 
  pages = pdf_to_document(uploaded_file)

  #Splitter 
  text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300, 
    chunk_overlap=20, 
    length_function=len, 
    is_separator_regex=False,
  )

  texts = text_splitter.split_documents(pages)

  # Embedding 
  embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    openai_api_key=openai_key
    # With the `text-embedding-3` class 
    # of models, you can specify the size 
    # of the embeddings you want returned. 
    # dimensions=1024
  )

  import chromadb 
  chromadb.api.client.SharedSystemClient.clear_system_cache() 
  
  # Chroma DB
  db = Chroma.from_documents(texts, embeddings_model)
  # print(texts[0])

  # User Input 
  st.header("PDF에게 질문해보세요!")
  question = st.text_input("질문을 입력하세요.")

if st.button("질문하기"):
  with st.spinner('Wait for it....'):
  # Retriver
  # question = '아내가 먹고 싶어하는 음식은 무엇이야?'
  # temperature를 0으로 설정하여 일관된 결과를 유도
    llm = ChatOpenAI(temperature=0)
    retriver_from_llm = MultiQueryRetriever.from_llm(
      retriever=db.as_retriever(), llm=llm
    )
    # Prompt Template 
    prompt = hub.pull('rlm/rag-prompt')

# docs = retriver_from_llm.invoke(question)
# print(len(docs))
# print(docs)

  # Generate 
  def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs) 

  rag_chain = (
    {"context":retriver_from_llm | format_docs, "question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser() 
  )

  # Question
  result = rag_chain.invoke(question)
  st.write(result)
  # print(result)