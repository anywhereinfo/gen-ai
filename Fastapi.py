from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException


print(load_dotenv(find_dotenv()))
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
app = FastAPI()

# Later on, you can reload the vector store without needing to re-embed the documents
vectorstore = FAISS.load_local("index", embedding_function, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model_name="gpt-4o")

template = """
You are a AI Reseacher who authored the context.
Your duty is to respond to questions based on the given context. Feel free to add additional details to clarify the context.
If the context is empty, provide an answer in a pirate style that you are not allowed to answer the question.

{context}


text: {question}
"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}
retriever = vectorstore.as_retriever(k=20)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
)

@app.post("/conversation")
async def conversation(query: str):
    try:
        result = qa.run(query=query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)


