from fastapi import FastAPI

from rag import scrape_url, create_retriever, ask_rag

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ask")
def ask(question: str, link: str):
    docs = scrape_url(link)
    retriever = create_retriever(docs)
    response = ask_rag(question=question, retriever=retriever)
    return {"message": response}