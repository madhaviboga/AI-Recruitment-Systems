import streamlit as st
from db_manager import validate_user

def login_page():
    # Center the login form using Streamlit form layout
    with st.form(key="login_form"):
        # Title
        st.title("Login Page")

        # Email and Password inputs
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        # Submit button inside the form
        login_button = st.form_submit_button("Login")

        # Handling form submission
        if login_button:
            user = validate_user(email, password)
            if user:
                # Set session state to user_home and store user details
                st.session_state["page"] = "user_home"
                st.session_state["user"] = user  # Store user info (e.g., name, email)
                st.session_state["user_tab"] = "Loan Page"  # Default tab after login
                st.experimental_rerun()
            else:
                st.error("Invalid Email or Password!")
