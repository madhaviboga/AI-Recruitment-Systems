import streamlit as st
import re
from db_manager import register_user

def register_page():
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url('https://img.freepik.com/premium-photo/school-education-background-banner-top-view_584012-3564.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-color: rgba(255, 255, 255, 0.6); /* Add a semi-transparent overlay */
        background-blend-mode: overlay; /* Blend the image with the overlay */

    }
    </style>
    """,
    unsafe_allow_html=True
    )
    # Center the registration form container using Streamlit form layout
    with st.form(key="register_form"):
        # Title
        st.title("Student Registration Form")
        # Form Fields
        col1, col2 = st.columns(2)
        name = col1.text_input("Name")
        email = col2.text_input("Email")
        regd_no = col1.text_input("Registration Number")
        branch = col2.selectbox("Branch", ["CSE", "ECE", "EEE", "MECH", "CIVIL", "IT", "AIDS", "BIO-TECH", "CHEMICAL", "ARCHITECTURE"])
        student_type = col1.selectbox("Student Type", ["Regular", "Lateral Entry", "Part-Time"])
        course=col2.selectbox("Course",["B.Tech","M.Tech","PhD","MBA","MCA","B.Sc","M.Sc","B.A","M.A","B.Com","M.Com"])
        college_name=col1.text_input("College Name")
        student_image = col2.file_uploader("Upload Student Image", type=["jpg", "jpeg", "png"])
        col1, col2 = st.columns(2)
        year_of_study = col1.selectbox("Year of Study", [1, 2, 3, 4])
        mobile_no = col2.text_input("Mobile Number")
        col1, col2 = st.columns(2)
        password = col1.text_input("Password", type="password")
        retype_password = col2.text_input("Retype Password", type="password")

        # Submit Button inside the form
        register_button = st.form_submit_button("Register",type='primary')

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
                if register_user(name, email, regd_no, branch, student_type, course, college_name, student_image, year_of_study, mobile_no, password):
                    st.success("Registration Successful!")
                else:
                    st.error("Email already exists!")
