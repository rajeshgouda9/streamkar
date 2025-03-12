##🚀 StreamKar AI Customer Support Bot

#📥 Setup Instructions

1. Clone the Repository
2. Set Google API Key - in settings.py update your googleapi key
3. Install dependencies - pip install -r requirements.txt
4. Start the Server - python main.py

# API Endpoints

1. Add FAQ
   To add an FAQ to the database and update the FAISS index, send a POST request:
   Endpoint:
   POST /add_faq

   Request Body:
   {
   "question": "How to start a stream on StreamKar?",
   "answer": "To start a stream, go to the Stream button and follow the on-screen instructions."
   }

2. Chat with the Bot (WebSocket)
   To start a chat session, connect to the WebSocket:
   Endpoint:
   ws://localhost:8000/chat/{user_id}
   Request Example:
   {
   "preferred_language": "Hindi",
   "question": "what is your name"
   }
