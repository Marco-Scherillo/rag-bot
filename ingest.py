from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

def split_file(path): #splits files in to ~500 chars and loads chromadb
    with open(path, encoding='utf-8') as file:
        content = file.read()
        context = []
        curr = ""
        for para in content.split('\n\n'):
            if len(curr) + len(para) > 500:
                context.append(curr)
                curr = para
            else:
                curr += para
        if len(curr) > 0:
            context.append(curr)
        return context

def get_chuncks(chuncks):
    context = []
    for chunck in chuncks:
        context.extend(chunck[0])
    return context

def get_metadatas(chuncks):
    metas = []
    for chunk in chuncks:
        n = len(chunk[0])
        metas.extend([{"source" : chunk[1]}] * n)
    return metas



folder = Path('./docs')
all_chuncks = []
model = SentenceTransformer("all-MiniLM-l6-v2")
for file in folder.iterdir():
    if file.is_file():
        print(f"Processing {file.name}")
        all_chuncks.append((split_file(file), file.name))

flat_list = get_chuncks(all_chuncks)
embedings = model.encode(flat_list)
client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection("docs")
collection.add(
    documents=flat_list,
    embeddings=embedings,
    ids= [f"chunk_{i}" for i in range(len(flat_list))],
    metadatas= get_metadatas(all_chuncks)
)