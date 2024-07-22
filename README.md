# AskPDF - A Streamlit App for Chatting with PDFs

This project offers a user-friendly interface to ask questions about the content of uploaded PDFs. It leverages Google Generative AI (GenAI) and Langchain libraries to achieve this functionality.

# Features:

-Extracts text from uploaded PDFs.
-Chunks the extracted text for efficient processing.
-Creates a vector store using GenAI embeddings for document retrieval.
-Implements a chatbot interface powered by GenAI to answer user questions based on the PDF content.

# Requirements:

Python 3.x
Streamlit (pip install streamlit)
langchain (pip install langchain)
langchain-google-genai (pip install langchain-google-genai)
langchain-community (pip install langchain-community)
PyPDF2 (pip install PyPDF2)
dotenv (pip install python-dotenv)

# Setup:

1) Clone this repository.

2) Install required libraries using pip install -r requirements.txt 

3) Create a .env file in the project root directory and set the following environment variable:

4) GOOGLE_API_KEY: Your Google Cloud project's API key with access to GenAI services.
   
# Usage:

Run the application using streamlit run main.py.
Upload your PDF documents in the Streamlit sidebar.
Click "Submit" to process the PDFs and create the internal knowledge base.
Once processing is complete, enter your questions in the "Ask your questions..." text box.
The application will analyze the PDFs and leverage the GenAI chatbot to provide answers based on the context.
