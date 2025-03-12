from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from rag_model import handle_with_llm, initialize_faq, update_faq_vectorstore
import logging
from pydantic import BaseModel
from models import Chat, SessionLocal, FAQ
from googletrans import Translator, LANGUAGES

logging.basicConfig(level=logging.INFO)

app = FastAPI()

translator = Translator()

class FAQRequest(BaseModel):
    question: str
    answer: str

def save_chat(user_id, input_text, response_text):
    try:
        session = SessionLocal()
        new_chat = Chat(
            user_id=user_id,
            input=input_text,
            response=response_text
        )
        session.add(new_chat)
        session.commit()
        session.refresh(new_chat)
        print(f"Chat saved with ID: {new_chat.id}")
    except Exception as e:
        session.rollback()
        print(f"Error saving chat: {e}")
    finally:
        session.close()
        
def insert_faq(question, answer):
    session = SessionLocal()
    try:
        new_faq = FAQ(question=question, answer=answer)
        session.add(new_faq)
        session.commit()
        logging.info("New FAQ added successfully.")
        # Refresh FAISS index after adding FAQ
        update_faq_vectorstore()
    except Exception as e:
        logging.error(f"Error inserting FAQ: {e}")
        session.rollback()
    finally:
        session.close()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post('/add_faq', tags=['FAQ'])
def add_faq(faq: FAQRequest):
    session = SessionLocal()
    try:
        # Insert into database
        new_faq = FAQ(question=faq.question, answer=faq.answer)
        session.add(new_faq)
        session.commit()

        # Refresh FAISS index after inserting
        update_faq_vectorstore()

        logging.info(f"New FAQ added: {faq.question}")
        return {"status": "success", "message": "FAQ added successfully"}

    except Exception as e:
        session.rollback()
        logging.error(f"Error adding FAQ: {e}")
        raise HTTPException(status_code=500, detail="Failed to add FAQ")

    finally:
        session.close()

@app.websocket("/chat/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    logging.info(f"WebSocket connection open for user: {user_id}")

    try:
        while True:
            data = await websocket.receive_text()
            logging.info(f"Received user query: {data}")
            data = eval(data)
            inp = data['question']
            detected_lang = translator.detect(inp).lang
            logging.info(f"Detected language: {detected_lang}")
            preffered_language = data['preffered_language']
            
            if detected_lang != 'en':
                # Translate to English
                translated_data = translator.translate(inp, src=detected_lang, dest='en').text
                logging.info(f"Translated input to English: {translated_data}")
            else:
                translated_data = inp

            # Use retriever + LLM to generate a response
            response = handle_with_llm(translated_data,preffered_language)

            logging.info(f"Sending response to user: {response}")
            save_chat(user_id, data, response) # this should be in bg
            await websocket.send_text(response)

    except Exception as e:
        logging.error(f"Error in WebSocket connection: {e}")

    finally:
        logging.info(f"WebSocket connection closed for user: {user_id}")
        await websocket.close()

initialize_faq()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
