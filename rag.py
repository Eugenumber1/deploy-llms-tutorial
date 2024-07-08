import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

prompt = hub.pull("rlm/rag-prompt")



def scrape_url(url: str) -> list:
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict()
    )
    docs = loader.load()
    return docs

def create_retriever(docs: list) -> VectorStoreRetriever:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    return retriever




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_rag(question: str, retriever: VectorStoreRetriever):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    return answer




