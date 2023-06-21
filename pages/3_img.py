import os
import openai
from PIL import Image
import streamlit as st

api_key = st.sidebar.text_input("API Key", type="password")

def openai_image(prompt):
    openai.api_key = api_key
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256"
    )
    image_url = response['data'][0]['url']
    return image_url

st.title("Image Generator")

st.write("Enter text to generate captivating visuals with our user-friendly interface.")

text_input = st.text_input("Text Input", "")

if text_input.strip() != "":
    if api_key.strip() != "":
        with st.spinner("Loading... 💫"):
            image_url = openai_image(text_input)
            st.image(image_url, caption='Generated by OpenAI')
    else:
        st.warning("Please enter an API key! ⚠")
else:
    st.warning("Please enter text to generate an image! ⚠")
