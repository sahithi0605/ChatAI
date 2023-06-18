# Import necessary libraries
import streamlit as st
import os
import openai
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from translate import Translator
import asyncio
import io
import time


languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi",
    "Telugu": "te",
    # Add more languages as needed
}

# Set Streamlit page configuration
st.set_page_config(page_title='MemoryBot', layout='wide')
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


def translate_text(text, target_lang):
    translator = Translator(to_lang=target_lang)
    translation = translator.translate(text)
    return translation


# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = " "


def main():
# Set up sidebar with various options
    with st.sidebar.expander("🛠️ Settings ", expanded=False):
    # Option to preview memory store
        MODEL = st.selectbox(label='Model', options=['text-davinci-003','text-davinci-002', 'gpt-3.5-turbo'])
        K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

# Set up the Streamlit app layout
    st.title("Interactive Chat")
    st.write(" Engage with the AI model, ask questions, seek information, or receive personalized responses")

# Ask the user to enter their OpenAI API key
# 
    api_key = st.sidebar.text_input("API Key", type="password")

# Session state storage would be ideal
    if api_key:
    # Create an OpenAI instance
        llm = OpenAI(temperature=0,
                    openai_api_key=api_key, 
                    model_name=MODEL, 
                    max_tokens=2048,
                    verbose=False) 

    # Create a ConversationEntityMemory object if not already created
        if 'entity_memory' not in st.session_state:
                st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
        
        # Create the ConversationChain object with the specified configuration
        Conversation = ConversationChain(
                llm=llm, 
                prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                memory=st.session_state.entity_memory
            )  
    else:
        st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.stop()


# Add a button to start a new chat
    st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Get the user input
    user_input = get_text()
    selected_lang = st.selectbox("Select Language", list(languages.keys()))

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
    if user_input:
        output = Conversation.run(input=user_input)  
        st.session_state.past.append(user_input)  
        st.session_state.generated.append(output)   
# Allow to download as well
    download_str = []
# Display the conversation history using an expander, and allow the user to download it
    with st.expander("Conversation", expanded=True):
        # for i in range(len(st.session_state['generated'])-1, -1, -1):
        #     st.info(st.session_state["past"][i],icon="🧑")
        #     translated_text = translate_text(st.session_state["generated"][i], languages[selected_lang])
        #     st.success(translate_text, icon="🤖")
        #     download_str.append(st.session_state["past"][i])
        #     download_str.append(st.session_state["generated"][i])
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                
            translated_text = translate_text(st.session_state["generated"][i], languages[selected_lang])
            message(translated_text, key=str(i), avatar_style="bottts")
                # else:
                #     message(st.session_state["generated"][i], key=str(i), avatar_style="robot")    
                
            download_str.append(st.session_state["past"][i])
            download_str.append(st.session_state["generated"][i])
       
    # Can throw error - requires fix
        download_str = '\n'.join(download_str)
        if download_str:
            st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
    for i, sublist in enumerate(st.session_state.stored_session):
            with st.sidebar.expander(label= f"Conversation-Session:{i}"):
                st.write(sublist)

# Allow the user to clear all stored conversation sessions
    if st.session_state.stored_session:   
        if st.sidebar.checkbox("Clear-all"):
            del st.session_state.stored_session
            
main()