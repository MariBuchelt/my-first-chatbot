from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from fastapi import FastAPI
from langserve import add_routes
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

#Get PDF Data
file_path = (
    "./my-chatbot/AIEngineer.pdf"
)
loader = PyPDFLoader(file_path)
pages = loader.load_and_split() 
text_splitter = RecursiveCharacterTextSplitter()
documents_pdf = text_splitter.split_documents(pages)

#Get Website Data
loader = WebBaseLoader(["https://www.promtior.ai","https://www.promtior.ai/service"])
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents_web = text_splitter.split_documents(docs)

#Merge pdf data with website data 
all_documents = documents_pdf + documents_web

embeddings = OpenAIEmbeddings()

#Create vectorstores
vector = FAISS.from_documents(all_documents, embeddings)
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = ConversationalRetrievalChain.from_llm(llm, retriever)

#App definition
app = FastAPI(
    title="My first Chatbot",
    version="1.0",
    description="This is a simple chatbot about Promtior information.",
)

add_routes(
    app, 
    chain,
    path="/agent",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

