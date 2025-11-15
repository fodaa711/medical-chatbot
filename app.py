from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

print("🔄 Loading environment variables...")
load_dotenv()

print("🔑 Checking API keys...")
if not os.environ.get("PINECONE_API_KEY"):
    print("❌ ERROR: PINECONE_API_KEY not found in .env file!")
    exit(1)
if not os.environ.get("GROQ_API_KEY"):
    print("❌ ERROR: GROQ_API_KEY not found in .env file!")
    exit(1)
print("✅ API keys loaded successfully")

print("📦 Importing dependencies...")
try:
    from langchain_groq import ChatGroq
    from langchain_pinecone import PineconeVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from pinecone import Pinecone
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install flask langchain-groq langchain-pinecone python-dotenv pinecone sentence-transformers langchain-community")
    exit(1)

app = Flask(__name__)

print("🔌 Connecting to Pinecone...")
try:
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    INDEX_NAME = "medical-chatbot"
    index = pc.Index(INDEX_NAME)
    print("✅ Pinecone connected successfully")
except Exception as e:
    print(f"❌ Pinecone connection failed: {e}")
    exit(1)

print("🧠 Loading embeddings model...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embeddings model loaded")
except Exception as e:
    print(f"❌ Embeddings loading failed: {e}")
    exit(1)

print("📊 Initializing vectorstore...")
try:
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    print("✅ Vectorstore initialized")
except Exception as e:
    print(f"❌ Vectorstore initialization failed: {e}")
    exit(1)

print("🤖 Initializing Groq LLM...")
try:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    print("✅ Groq LLM initialized")
except Exception as e:
    print(f"❌ Groq initialization failed: {e}")
    exit(1)

def ask_medical_question(user_question, top_k=10):
    """Process user question and return medical answer"""
    try:
        # Rewrite unclear user question
        rewrite_prompt = f"""
        Rewrite the following question into a clear, concise medical query for retrieval:
        User question: "{user_question}"
        Clear question:
        """
        clear_question = llm.invoke(rewrite_prompt).content
        
        # Retrieve top-k relevant embeddings from Pinecone
        docs = vectorstore.similarity_search(clear_question, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # System prompt
        system_prompt = """
        You are a professional medical assistant.
        Provide accurate answers based only on the provided context.
        If the context does not contain the answer, say 'I don't know'.
        Explain medical terms in a patient-friendly way.
        """
        
        # Combine system prompt, context, and user question
        final_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_question}\nAnswer:"
        
        # Get response from LLM
        response = llm.invoke(final_prompt)
        return response.content
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route("/")
def index():
    """Render the chat interface"""
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    """Handle chat messages"""
    msg = request.form["msg"]
    
    if not msg:
        return "Please enter a message."
    
    # Get response from medical chatbot
    response = ask_medical_question(msg)
    return response

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Flask server...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
