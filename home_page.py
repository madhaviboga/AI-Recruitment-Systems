import streamlit as st
from login_page import login_page
def home_page():
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area with transparency */
        .main {
            background-image: url('https://www.orangemantra.com/blog/wp-content/uploads/2023/12/ai-in-education.png');
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

    st.markdown(
        """
        <div style="text-align: center; color: white;">
            <h1 style="color: white; font-size: 80px; text-align: center;">AI Recruitment Systems</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    #add image
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://cdni.iconscout.com/illustration/premium/thumb/robot-learning-illustration-download-in-svg-png-gif-file-formats--ai-study-artificial-intelligence-advance-technology-machine-pack-science-illustrations-4618518.png?f=webp", width=500, height=400>
        </div>
        """,
        unsafe_allow_html=True
    )

