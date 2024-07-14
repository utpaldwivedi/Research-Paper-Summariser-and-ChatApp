import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(multiple_pdfs):
    text=""
    for pdf in multiple_pdfs:
        pdf_reader=PdfReader(pdf)
        for pages in pdf_reader.pages:
            text+=pages.extract_text()
    return text

def create_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=2000)
    chunks=text_splitter.split_text(text)
    return chunks

def create_text_embeddings(chunks):
    embedding=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store=FAISS.from_texts(chunks,embedding=embedding)
    vector_store.save_local("gemini_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    don't provide the wrong answer in any case\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def response_to_query(query,chunks=None):
    embedding=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    relevant_chunks=None
    if chunks: 
        relevant_chunks = [Document(page_content=chunk) for chunk in chunks]
    else:    
        db=FAISS.load_local("gemini_index",embeddings=embedding,allow_dangerous_deserialization=True)
        relevant_chunks=db.similarity_search(query)
    
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents":relevant_chunks, "question": query}, return_only_outputs=True)
    return response["output_text"]

def main():
    flag=False
    summary=None
    with st.sidebar:
        st.markdown("# 🤗A LLM Summariser \& Chat App")
        st.write("")
        st.write("1. Upload your pdf files to get the summary and ask questions.")
        st.write("2. Then click submit and process.")
        st.write("3. Once processed , check results on the right")
        files=st.file_uploader("Upload your files here",type="pdf",accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if files:
                    text=extract_text_from_pdf(files)
                    chunks=create_text_chunks(text)
                    create_text_embeddings(chunks)
                    summary=response_to_query("Give me the summary of entire PDF document",chunks=chunks)
                    flag=True
                    st.success("Done")
                else:
                    st.error("Please provide a PDF")

    st.markdown("# Chat with PDF🤔")
    user_input=st.text_area("Ask a question from PDF files")

    if user_input:
        res_gemini=response_to_query(user_input)
        st.write(res_gemini)
    elif flag:
        st.write(summary)

    

if __name__=="__main__":
    main()