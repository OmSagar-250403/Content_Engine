import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks


# Function to create embeddings and vector store
def get_vectorstore(text_chunks):
    try:
        # Create embeddings using GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # The FAISS vector store automatically handles the embedding creation
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local("faiss_index")  # Save the FAISS index locally
        st.success("FAISS index created and saved successfully!")
        # Return the vectorstore
        return vectorstore

    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

# Function to handle user input
def handle_userinput(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history.append(response['chat_history'])
    st.write(response['answer'])

# Function to create conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Main function to run the app
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.header("Chat with Your PDF Files :books:")
    
    pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
    
    if st.button("Process"):
        if pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.chat_history = []  # Initialize chat history
                st.success("PDF processed successfully!")
            else:
                st.error("There was an error creating the vectorstore.")
        else:
            st.error("Please upload at least one PDF file.")
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
