import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient

import chromadb

# Import Task libraries
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

class ChromaCollectionCreator:
    def __init__(self, processor, embed_model):
        
        # Initializing the ChromaCollectionCreator with a DocumentProcessor instance and embeddings configuration.
        self.processor = processor      # This will hold the DocumentProcessor from Task 3
        self.embed_model = embed_model  # This will hold the EmbeddingClient from Task 4
        self.db = None                  # This will hold the Chroma collection

    def as_retriever(self):
      
      # Retrieve for the Chroma collection created within this class.
      
      if self.db:
          # If the Chroma collection exists, return its built-in retriever
          return self.db.as_retriever()
      else:
          # Raise an error if the Chroma collection hasn't been created
          return None  # Or handle it differently (e.g., raise an error)

    def create_chroma_collection(self):
        

        # Creating a Chroma collection from the documents processed by the DocumentProcessor instance.
        # Step 1: Check for processed documents
        if len(self.processor.pages) == 0:
            st.error("No documents found!", icon="🚨")
            return

        # Step 2: Split documents into text chunks
        # Use a TextSplitter from Langchain to split the documents into smaller text chunks
        # Define the separator, chunk size, and chunk overlap
        text_splitter = CharacterTextSplitter(
                            separator = "\n",  #  separator
                            chunk_size = 1000,  #  chunk size
                            chunk_overlap = 100,  # chunk overlap
                            length_function = len,
                            is_separator_regex=False
                            )

        # Split the processed documents into text chunks
        texts = text_splitter.split_documents(self.processor.pages)
        
        
        if texts is not None:
            st.success(f"Successfully split pages to {len(texts)} documents!", icon="✅")

        # Step 3: Create the Chroma Collection
        # Create a Chroma in-memory client using the text chunks and the embeddings model
        self.db = Chroma.from_documents(texts, self.embed_model)
        
        if self.db:
            st.success("Successfully created Chroma Collection!", icon="✅")
        else:
            st.error("Failed to create Chroma Collection!", icon="🚨")
    
    def query_chroma_collection(self, query) -> Document:
        
        # Query the created Chroma collection for documents similar to the query.

        if self.db:
            docs = self.db.similarity_search_with_relevance_scores(query)
            if docs:
                return docs[0]
            else:
                st.error("No matching documents found!", icon="🚨")
        else:
            st.error("Chroma Collection has not been created!", icon="🚨")

if __name__ == "__main__":
    processor = DocumentProcessor() # Initialize from Task 3
    processor.ingest_documents()
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "YOUR PROJECT ID",
        "location": "us-central1"
    }
    
    embed_client = EmbeddingClient(**embed_config) # Initialize from Task 4
    
    chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
    with st.form("Load Data to Chroma"):
        st.write("Select PDFs for Ingestion, then click Submit")
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            chroma_creator.create_chroma_collection()