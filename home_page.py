import streamlit as st
from login_page import login_page
def home_page():
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area with transparency */
        .main {
            background-image: url('https://www.shutterstock.com/image-photo/smart-ai-technology-system-human-600nw-2336385527.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.3); /* Add a semi-transparent overlay */
            background-blend-mode: overlay; /* Blend the image with the overlay */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    col1,col2,col3 = st.columns([1,1,1])
    col2.image("https://pngimg.com/d/ai_PNG4.png")

    
