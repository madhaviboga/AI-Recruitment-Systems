import streamlit as st

def home_page():
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>AI Recruitment Systems</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Center the image
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://news.microsoft.com/wp-content/uploads/prod/sites/388/2018/05/limitedexperience_hero_social.gif" style="max-width: 50%;">
        </div>
        """,
        unsafe_allow_html=True
    )
