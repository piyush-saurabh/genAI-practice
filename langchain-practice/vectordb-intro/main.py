import os
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone  # pinecone client

from langchain import VectorDBQA, OpenAI


if __name__ == "__main__":
    print("Hello Langchain")

    # take environment variables from .env.
    # usage: print(os.environ.get("OPENAI_API_KEY"))
    load_dotenv()

    # Provide the full path of the text file
    loader = TextLoader(
        "/Users/crispler/Documents/study/AI/vectordb-intro/mediumblogs/mediumblog1.txt"
    )

    # Convert the text file into a command format called document
    document = loader.load()

    # Split the long document into smaller size
    # TODO: fine tune chunk_size and chunk_overlap based on the LLM response
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    # This will be the count of the number of vector embeddings
    # number of chunks = number of vectors in the vector db
    print(len(texts))

    # Initialize the embedding
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Initialize Pinecode database
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"), environment="us-west4-gcp-free"
    )

    # Conver the splitted texts into embeddings using OpenAI and save it into vector database
    # Index is like a table for storing vectors
    # Create index in Pinecone with specific 'dimension' and 'metrics'
    # Dimension: Number of elements we have in the vector embeddings. google "open ai embedding dimension"
    # Metrics: cosine, dotproduct, eucledian (select this)
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embedding-index"
    )

    # Initialize the Question Answering chain
    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        vectorstore=docsearch,
        return_source_documents=False,
    )

    # Query
    # query = "What is the vector database? Explain it for a beginner"
    query = "who is the author of this blog?"

    result = qa({"query": query})

    print(result)
