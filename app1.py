import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

URL = "https://www.sec.gov/Archives/edgar/data/1652044/000165204423000016/goog-20221231.htm"
CHROMA_PATH = "chroma_db"

docs = WebBaseLoader(URL).load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

query = "what are the top risks mentioned in the document?"
top = db.similarity_search(query, k=5)
context = "\n\n".join(d.page_content for d in top)

prompt = ChatPromptTemplate.from_template(
    "Answer using only the context:\n{context}\n\nQuestion: {question}"
).format(context=context, question=query)

ans = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0).invoke(prompt).content
print(ans)
