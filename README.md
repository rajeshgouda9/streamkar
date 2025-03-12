##🚀 StreamKar AI Customer Support Bot

#Prerequisites

1. you should have a postgres db running on local you can change the cred in models.py for the database
2. create a database called streamkar

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

## model Selection

1. I am using gemeni because i dont have openai creds
2. Understand and classify user queries we can use a zeroshot model to check the input type from selected label and redirect the request accordingily

3. we can also train a few shot model for the same the response time would be around 50ms for zero or fewshot model in a gpu machine

4. we can use google translate api but we can also ask llm to respond in a user preferred language directly and if user inp is not in english then convert it to english

5. Redis for caching frequently asked queries. pending

6. Celery for background tasks (e.g., training, logging). pending
