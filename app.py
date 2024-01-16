"""
##########################################################
This is project can be used only for demo purposes 
Author: Harini Datla
############################################################
s"""
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# LOADING THE GOOGLE API KEY
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
   """Extracts text from multiple PDF documents and combines it into a single string.

   Args:
       pdf_docs (list): A list of paths to PDF documents.

   Returns:
       str: The combined text extracted from all the PDF documents.

   Raises:
       ImportError: If the required library 'PyPDF2' is not installed.

   Example:
       >>> pdf_text = get_pdf_text(['document1.pdf', 'document2.pdf'])
       >>> print(pdf_text)
   """

   try:
       from PyPDF2 import PdfReader
   except ImportError:
       raise ImportError("Please install the PyPDF2 library to use this function.")

   text = ""
   for pdf in pdf_docs:
       pdf_reader = PdfReader(pdf)
       for page in pdf_reader.pages:
           text += page.extract_text()
   return text

import logging

def get_text_chunks(text):
    """Splits a large text into overlapping chunks of manageable size.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of text chunks, each chunk being a string.

    Raises:
        ValueError: If the input text is empty.

    Logs:
        - Warning if empty text is encountered.
        - Information about successful chunking.
        - Error if any exceptions occur.
    """

    logger = logging.getLogger(__name__)  # Get a logger for this module

    try:
        if not text:
            logger.warning("Input text is empty.")
            raise ValueError("Input text cannot be empty.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        logger.info("Text successfully chunked into %d chunks.", len(chunks))
        return chunks

    except ValueError as e:
        logger.error("Error: %s", e)
        return []


def get_vector_store(text_chunks):
    """Creates a vector store (index) from text chunks using Faiss and Google Generative AI embeddings.

    """

    logger = logging.getLogger(__name__)  # Get a logger for this module

    logger.info("Creating embeddings with Google Generative AI model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    logger.info("Building Faiss vector store...")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    logger.info("Saving vector store locally...")
    vector_store.save_local("faiss_index")

    logger.info("Vector store creation complete.")
    return vector_store

def get_conversational_chain():
    """Creates a conversational chain for question-answering using a Google AI generative model.

    Args:
        None

    Returns:
        A loaded conversational chain object for question-answering.

    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context, just say, "answer is not available in the provided pdf".
    Don't provide the wrong context.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Processes a user's question and provides a response using a conversational AI model.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        None (prints and displays the response)

    Steps:
        1. Loads a Google Generative AI embeddings model.
        2. Loads a previously saved Faiss vector store (index).
        3. Finds similar documents to the user's question using Faiss.
        4. Creates a conversational chain for question-answering.
        5. Runs the chain with the question and similar documents as context.
        6. Prints and displays the generated response using Streamlit.

    """

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config("Guassian Search")
    st.header("Guassian AI")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()








