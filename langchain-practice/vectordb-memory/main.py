import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


if __name__ == "__main__":
    print("Hello Langchain")

    # take environment variables from .env.
    # usage: print(os.environ.get("OPENAI_API_KEY"))
    load_dotenv()
    
    # Get the pdf file path
    pdf_path = "/Users/crispler/Documents/study/AI/vectordb-memory/docs/ssrf.pdf"

    # Load the pdf
    loader = PyPDFLoader(file_path=pdf_path)
    pdf_doc = loader.load()

    # Chunkify the document so that it does not exceeds LLM token limit
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    document = text_splitter.split_documents(documents=pdf_doc)

    # Create the embeddings using open ai
    embeddings = OpenAIEmbeddings()

    # # use openai for embedding and store it in FAISS
    # # This is stored in RAM
    # # This is required for the 1st time to create the embeddings using OpenAI. After the embeddings are created and stored, we can comment this line
    # vectorstore = FAISS.from_documents(document, embeddings)

    # # Save this vector in local file system by providing the index name
    # # This is required for the 1st time. After the embeddings are created and stored, we can comment this line
    # vectorstore.save_local("faiss_index_ssrf")

    # Load the embeddings from local storage
    loaded_vectorstore = FAISS.load_local("faiss_index_ssrf", embeddings)

    # Initialize the Question Answering chain
    # chain_type = "stuff" means the result from the vectordb will be stuffed as the context along with the prompt 
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=loaded_vectorstore.as_retriever())

    # Query
    #query = "what is server side request forgery?"

    #query = "As an attacker, how can I find server side request forgery?"
    # response: "Attackers can uncover server-side request forgery vulnerabilities by using standard techniques such as partial request URLs, manipulating URL parameters, and monitoring traffic flow."

    # query = "How attacker find ssrf by manipulating URL parameters?"
    # response: "Attackers can recognize partial request URLs as a URL path and modify it, enabling the server to make malicious requests."

    # query = "Can you explain in details how we can find ssrf by manipulating URL parameters?"
    # response: "Yes. Attackers can find SSRF vulnerabilities by manipulating URL parameters. They may use techniques such as partial request URLs, using case variation or URL obfuscation to obfuscate blacklisted strings, leveraging wildcard DNS services, confusing the URL parser with URL encoded characters, and using whitelist-based input filters.."

    # query = "What are the different categories of SSRF?"
    # response: "The two categories of SSRF are blind SSRF attacks and direct SSRF attacks. Blind SSRF attacks exploit vulnerabilities that allow attackers to issue a server-side request to a URL with no response reflected in the application's client-side response. Direct SSRF attacks involve the hacker tricking the web application into issuing a server-side request and obtaining the contents of the server-side response through the application's client-side response."

    # This response is not quite correct
    # query = "What are the different ways to bypass SSRF protection?"
    # response: "Some common approaches to bypassing SSRF protection include using alternative IP address representations, registering a domain name that resolves to a blacklisted target address, using a blacklist-based input filter, and using multi-factor authentication, role-based authorizations, or other rule-based security measures at the network perimeter."

    query = "What are different ways to protect our application from SSRF?"
    # response: Recommended practices to prevent SSRF vulnerabilities include disabling unused URL schemas, enforcing input sanitization and validation, implementing strict access controls and firewall policies, and using case variation or URL obfuscation to obfuscate blacklisted strings.



    # Run the chain with the query
    result = qa.run(query)

    print(result)



