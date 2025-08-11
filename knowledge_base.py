# File: knowledge_base.py
# Libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
import os
from dotenv import load_dotenv

# Set OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load documents
loader = DirectoryLoader('Documents/')
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Generate embeddings
embeddings = OpenAIEmbeddings()

# Create and persist vector store
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
vectorstore.persist()

print("Knowledge base prepared and stored in ./chroma_db")