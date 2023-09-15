from dotenv import load_dotenv
from backend.core import run_llm
from backend.core import run_llm_with_memory
import streamlit as st
from streamlit_chat import message

# function which takes the list of URLs and formats it
def create_sources_string(sources):
    if not sources:
        return ""
    sources_list = list(sources)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

# Creating state (local storage) for streamlit
# Local storage 1: History of questions/prompts
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

# Local storage 2: History of response
if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

# Creating memory for the chat. This will be required to ask question on the response from the chat history
# This chat history is passed on to the LLM as context
# Local storage 3: Memory for the chat
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [] 

# Create webpage title
st.header("Langchain Documentation Helper Bot")

# Textbox
prompt = st.text_input("Prompt", placeholder="Enter your question here...")

# If we have any question, run this
if prompt:
    # Show the loading icon with the message while fetching response from LLM
    with st.spinner("Finding answer..."):

        # LLM without memory
        #generated_response = run_llm(query=prompt)

        # Run the LLM with the history. This will be the 'memory'
        generated_response = run_llm_with_memory(query=prompt, chat_history=st.session_state["chat_history"])

        # Get the list of URLs for all the source documents obtaind from similarity search from Pinecone
        # The URLs are converted into set to remove the duplicate URLs
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])

        # Format the response for displaying in the UI
        # Without memory
        # formatted_response = f"{generated_response['result']} \n\n {(create_sources_string(sources))}"

        # With memory
        formatted_response = f"{generated_response['answer']} \n\n {(create_sources_string(sources))}"

        # Save the state
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        
        # without memory
        # st.session_state["chat_answer_history"].append(prompt, generated_response['result'])

        # with memory
        st.session_state["chat_history"].append((prompt, generated_response['answer']))


# If we have some chat history, show it in the UI
if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip(st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)


