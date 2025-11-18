Medical Chatbot

Medical Chatbot is an AI-powered assistant designed to provide professional and well-structured medical information.
It uses advanced natural language processing, embeddings, and vector search to deliver accurate responses.

Features

Answers medical questions in a clear and professional way

Uses vector search for context-aware responses

Flask-powered backend with a simple web interface

Pinecone vector database for storing and retrieving medical knowledge

Hugging Face embeddings for semantic search

Groq/LLM integration for generating final responses

Easy to run locally and simple to deploy

Tech Stack

Backend:

Python

Flask

Groq API / LLM

Hugging Face Embeddings

Pinecone Vector Database

LangChain ecosystem

Frontend:

HTML

CSS

JavaScript

Installation and Setup
1. Clone the Repository
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot

2. Install Dependencies
pip install -r requirements.txt

3. Add Environment Variables

Create a .env file in the root directory and add:

GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_environment

How to Run the App in Visual Studio Code
1. Open the Project

Open Visual Studio Code

Select File → Open Folder

Choose the medical-chatbot folder

2. Run the Application

Open the integrated terminal in VS Code and run:

python app.py

3. Access the Chatbot

Open your browser and go to:

http://127.0.0.1:5000


The chatbot will now be running locally.

How It Works

The user enters a medical question.

The question is converted into embeddings using Hugging Face.

Pinecone retrieves the most relevant medical context.

The Groq LLM generates a safe, structured answer.

Flask returns the response to the user interface.

Safety Notice

This project is intended for educational and informational purposes only.
The chatbot does not provide medical diagnoses and should not replace consultations with licensed medical professionals.

Future Improvements

Add voice input support

Improve UI/UX design

Add symptom-based recommendations

Cloud deployment

Add caching layer for faster responses

Contributing

Contributions are welcome.
If you would like to suggest improvements or report issues, please open a GitHub issue or submit a pull request.
