import os
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone
# Import your API keys
import os
from config import PINECONE_API_KEY, GROQ_API_KEY
# Set them as environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load Existing index

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Define the index name
INDEX_NAME = "medical-chatbot" # Ensure INDEX_NAME is defined

# Connect to existing index
index = pc.Index(INDEX_NAME)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vectorstore using the initialized Pinecone index
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
