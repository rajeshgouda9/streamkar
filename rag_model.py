import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
import numpy as np
from models import FAQ, SessionLocal
from settings import GOOGLE_API_KEY
#TODO
#Understand and classify user queries we can use a zeroshot model to check the input type from selected label and redirect the request accordingily
# we can also train a few shot model for the same the response time would be around 50ms for zero or fewshot model in a gpu machine
# we can use google translate api but we can also ask llm to respond in a user preferred language directly and if user inp is not in english then convert it to english

#Redis for caching frequently asked queries. pending
# Celery for background tasks (e.g., training, logging). pending



# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Set API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# Similarity threshold for valid FAQ match
SIMILARITY_THRESHOLD = 0.50

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
FAISS_INDEX_PATH = "faiss_index"

def load_faq_from_db():
    """Load FAQ data from the database."""
    logging.info("Loading FAQ data from database...")
    session = SessionLocal()
    try:
        faqs = session.query(FAQ).filter(FAQ.is_active == True).all()
        faq_data = []
        if not faqs:
            doc = Document(page_content="Q: How to start a stream on StreamKar? A: To start a stream, go to the Stream button and follow the on-screen instructions.", metadata={})
            faq_data.append(doc)
        for faq in faqs:
            doc = Document(
                page_content=f"Q: {faq.question} A: {faq.answer}",
                metadata={"id": faq.id}
            )
            faq_data.append(doc)
        logging.info(f"Loaded {len(faq_data)} FAQs from database.")
        return faq_data
    finally:
        session.close()

def update_faq_vectorstore():
    """Update FAISS index with fresh FAQ data."""
    logging.info("Updating FAISS vectorstore with latest FAQ data...")
    faq_data = load_faq_from_db()
    if faq_data:
        vectorstore = FAISS.from_documents(faq_data, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        logging.info("FAISS index updated successfully.")
        global retriever, llm
        retriever, llm = create_chat_model()
    else:
        logging.warning("No FAQ data found to update FAISS index.")

def initialize_faq():
    """Initialize FAISS index from database."""
    logging.info("Initializing FAISS index...")
    update_faq_vectorstore()
    
# Load FAISS index
def load_faq():
    logging.info("Loading FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    logging.info("FAISS index loaded successfully.")
    return vectorstore

# Create Chat Model with Retriever + LLM fallback
def create_chat_model():
    logging.info("Creating Chat Model with Retriever + LLM fallback...")
    vectorstore = load_faq()
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Use cosine similarity by default
        search_kwargs={"k": 3}     # Retrieve top 3 most similar documents
    )

    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-pro-exp-02-05"
    )

    logging.info("Chat Model initialized.")
    return retriever, llm

# Function to calculate similarity score
def calculate_similarity_score(query_embedding, doc_embedding):
    similarity = np.dot(query_embedding, doc_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
    )
    return similarity

# Initialize model
initialize_faq()
retriever, llm = create_chat_model()

def handle_with_llm(query,preffered_language):
    logging.info(f"Received user query: '{query}'")
    
    # Step 1: Try to retrieve from FAQ
    logging.info("Searching for FAQ match...")
    docs = retriever.invoke(query)
    
    if docs:
        logging.info(f"Retrieved {len(docs)} document(s) from FAQ:")
         # Log the retrieved documents and their metadata (which may include scores)
        for i, doc in enumerate(docs):
            # Many retrievers return score in metadata
            logging.info(f"Doc {i + 1}: {doc.page_content}")
            
        # Process documents further...
        # For example, you might want to combine them into a context for the LLM
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 2: Send to LLM with context
        prompt = f'''
        You are StreamKar's customer support bot.
        Use the provided context to answer the user's query accurately.
        Format your response under 200 words, keeping it as concise as possible. You can engage in normal conversation with user.
        If you are not able to answer or need more information, or if the query should be redirected to a real customer support agent, just return 'REDIRECT'.

        <context_start>
        {context}
        <context_end>\n
        response preffered_language = {preffered_language}\n
        User asked: {query}
        Response:
        '''
        try:
            response = llm.invoke(prompt)
            if response and 'REDIRECT' not in response.content:
                logging.info(f"LLM Response: {response.content}")
                return response.content
            else:
                logging.info(f"LLM Response: {response.content}")
                logging.warning("LLM response is empty or REDIRECT.")
                return "Sending your request to the support team. Our team will contact you in some time."

        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return "An error occurred while processing your request."
    
    # If no FAQ match at all:
    logging.warning("No relevant FAQ match found. Falling back to LLM...")
    try:
        prompt = f'''
        You are StreamKar's customer support bot.
        Use the provided context to answer the user's query accurately.
        Format your response under 200 words, keeping it as concise as possible. You can engage in normal conversation with user.
        If you are not able to answer or need more information, or if the query should be redirected to a real customer support agent, just return 'REDIRECT'.

        User asked: {query}

        Response:
        '''
        response = llm.invoke(prompt)
        if response and 'REDIRECT' not in response.content:
            logging.info(f"LLM Response: {response.content}")
            return response.content
        else:
            logging.info(f"LLM Response: {response.content}")
            logging.warning("LLM response is empty or REDIRECT.")
            return "Sending your request to the support team. Our team will contact you in some time."
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return "An error occurred while processing your request."

if __name__ == "__main__":
    initialize_faq()
    for _ in range(4):
        query = input("Ask a question: ")
        response = handle_with_llm(query)
        print("Response:", response)
