import streamlit as st
import re
from db_manager import register_user

def register_page():
    # Center the registration form container using Streamlit form layout
    with st.form(key="register_form"):
        # Title
        st.title("Student Registration Form")
        # Form Fields
        col1, col2 = st.columns(2)
        name = col1.text_input("Name")
        email = col2.text_input("Email")
        regd_no = col1.text_input("Registration Number")
        branch = col2.selectbox("Branch", ["CSE", "ECE", "EEE", "MECH", "CIVIL", "IT", "AIDS", "BIO-TECH", "CHEMICAL", "MME", "MATHS", "PHYSICS", "CHEMISTRY", "ENGLISH", "MBA", "MCA", "PHD"])
        student_type = col1.selectbox("Student Type", ["Regular", "Lateral Entry", "Diploma", "PhD", "MBA", "MCA", "M.Tech", "M.Sc", "M.Phil", "Others"])
        student_image = col2.file_uploader("Upload Student Image", type=["jpg", "jpeg", "png"])
        year_of_study = st.slider("Year of Study", 1, 4, 1)
        col1, col2 = st.columns(2)
        password = col1.text_input("Password", type="password")
        retype_password = col2.text_input("Retype Password", type="password")

        # Submit Button inside the form
        register_button = st.form_submit_button("Register")

        # Handling form submission
        if register_button:
            # Validate email using regex
            email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
            if not re.match(email_regex, email):
                st.error("Invalid Email!")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long!")
            elif password != retype_password:
                st.error("Passwords do not match!")
            else:
                if register_user(name, email, password, regd_no, year_of_study, branch, student_type, student_image):
                    st.success("Registration Successful!")
                else:
                    st.error("Email already exists!")
