from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os 
from dotenv import load_dotenv
import json

load_dotenv()
model = SentenceTransformer("all-MiniLM-l6-v2")
client = Groq(api_key= os.getenv("GROQ_API_KEY"))
db = chromadb.PersistentClient(path="./chromadb")
collection = db.get_collection("docs")


question = input("Ask a question: ")

embedding = model.encode([question]).tolist()


results = collection.query(
    query_embeddings=embedding,
    n_results=3
)

context = results["documents"][0]

prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}
"""

responce = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role" : 'user', "content" : prompt}
    ]
)
print(responce.choices[0].message.content)