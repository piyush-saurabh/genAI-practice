import os
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA # used for chat without memory
from langchain.chains import ConversationalRetrievalChain # used for chat with memory
from langchain.vectorstores import Pinecone

import pinecone

# Load the environment variables
load_dotenv()

# Initialize pinecone client
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

PINECONE_INDEX = "langchain-docs-index"


# Run the LLM for Q&A without memory
def run_llm(query):
    embeddings = OpenAIEmbeddings()

    # Get the vector db instance which we will use in the chain
    docsearch = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX, embedding=embeddings
    )

    # Initialize the LLM
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Create the QA chain
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    # Obtain the answer by invoking the QA chain
    answer = qa({"query": query})
    return answer

# run the LLM for Q&A with memory
def run_llm_with_memory(query, chat_history):
    embeddings = OpenAIEmbeddings()

    # Get the vector db instance which we will use in the chain
    docsearch = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX, embedding=embeddings
    )

    # Initialize the LLM
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Create the QA chain with memory
    qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True)

    # Obtain the answer by invoking the QA chain along with chat history
    answer = qa({"question": query, "chat_history": chat_history})
    return answer

# Snippet for testing
if __name__ == "__main__":
    #print(run_llm(query="What is Retrieval QA chain?"))
    print(run_llm(query="What is langchain?"))
