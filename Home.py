import streamlit as st

# Set the page layout to wide
st.set_page_config(layout="wide")
# CSS styles
css_styles = """
<style>
/* Add your CSS styles here */
.navbar {
  display: flex;
  justify-content: center;
  margin-bottom: 20px;
}

.navbar a {
  margin: 0 10px;
  padding: 30px;
  color: #FFFFFF;
  background-color: #FF4B4B;
  text-decoration: none;
  border-radius: 20px;
}

.navbar a:hover {
  background-color: #FF4B4B;
}

.page-content {
  padding: 20px;
}

</style>
"""

# Render the CSS styles
st.markdown(css_styles, unsafe_allow_html=True)

# Streamlit app content
st.title("What can be done !")

# Navigation menu
page = st.session_state.get("page", "Home")
pages = ["Interactive Chat", "PDF Abstractor", "Image Generator"]


navbar_html = """
<div class="navbar">
  <a href="/chat" id="Interactive chat" class="page-link">Interactive chat</a>
  <a href="/pdf" id="PDF Abstractor" class="page-link">PDF Abstractor</a>
  <a href="/img" id="Image Generator" class="page-link">Image Generator</a>
</div>
"""

st.markdown(
    """
    <style>
    .navbar {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .navbar .page-link {
        margin: 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(navbar_html, unsafe_allow_html=True)


# Add some text content
st.header("About")
st.write("Experience the power of AI as you engage in captivating chat conversations, effortlessly summarize complex PDF documents, and unleash your creativity with an image generator that brings your visions to life.")

st.header("Features")
st.write("Here are  features of this interface:")
st.write("Conversational Translator ")
st.write("Quering and Summarizing the pdf")
st.write("Image generation system")


# JavaScript code for page navigation
js_code = """
<script>
const links = document.querySelectorAll('.page-link');

links.forEach(link => {
  link.addEventListener('click', (event) => {
    event.preventDefault();
    const pageId = link.getAttribute('id');
    const nextPage = pageId.charAt(0).toUpperCase() + pageId.slice(1).toLowerCase();
    Streamlit.setSessionState({page: nextPage});
  });
});
</script>
"""

st.markdown(js_code, unsafe_allow_html=True)
