import os
from dotenv import load_dotenv
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone


# This function takes the document > converts it into embeddings > store it in pinecone
def ingest_docs():
    # Load the document from local file system
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest")
    raw_document = loader.load()
    print(f"loaded {len(raw_document)} documents")

    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )

    # Split the documents into chunks
    documents = text_splitter.split_documents(documents=raw_document)
    print(f"splitted {len(documents)} documents")

    # Update the document metadata to show the exact url
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} documents into Pinecone")

    # Create embeddings and upload to pinecone
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(
        documents=documents, embedding=embeddings, index_name="langchain-docs-index"
    )

    print("Added all the embeddings to Pinecone DB")


if __name__ == "__main__":
    print("Hello Langchain")
    # take environment variables from .env.
    # usage: print(os.environ.get("OPENAI_API_KEY"))
    load_dotenv()

    # Initialize pinecone client
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
    )

    ingest_docs()
