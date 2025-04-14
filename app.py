from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import logging
import google.generativeai as genai
from functools import lru_cache
from fastapi.middleware.gzip import GZipMiddleware
import time
from openai import OpenAI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
client = OpenAI()

# Configure the Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI
app = FastAPI()
# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

def get_embedding(text: str):
    """Cached version of embedding generation"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002",
    )
    return response.data[0].embedding

def find_relevant_chunk(query: str, namespacee:str):
    """Finds the most relevant chunk using cached embeddings from Pinecone."""
    try:
        # Use cached embedding generation
        query_embedding = get_embedding(query)
        query_results = index.query(namespace=namespacee,vector=query_embedding, top_k=3,include_metadata=True)
        if query_results['matches']:
            match = query_results['matches'][0]
            text = match['metadata']['text']  
            return text
        else:
            return "Sorry, I couldn't find anything relevant."
    except Exception as e:
        logger.error(f"Error generating embeddings or querying Pinecone: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve relevant data.")

def generate_response(query: str, number: str):
    """Generates a chatbot response using groq API."""
    start_time = time.time()

    if query.lower() in ["exit", "bye"]:
        return "Thank you for using the service. Goodbye!"

    selected_chunk = find_relevant_chunk(query,number)
    print(selected_chunk)
    prompt = f"Context: {selected_chunk}\n\nUser: {query}\nChatbot:"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error generating response from Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

@app.post("/process-text/")
async def process_text(request: Request):
    """
    Endpoint that handles streaming both input and output
    """
    print("Request received")
    print("Headers:", dict(request.headers))
    
    try:
        body = await request.json()
        query = body.get("text", "")
        phone_number = body.get("phone_number", "")
        print(f"Received query: {query} for {phone_number}")
        
        response = generate_response(query,phone_number)
        print(f"Generated response: {response}")
        return JSONResponse(content={"response": response})
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
