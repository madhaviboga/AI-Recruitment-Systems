import streamlit as st
from db_manager import validate_user,update_otp,fetch_otp
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
def send_alert_email(to_email, subject, message, from_email, from_password):
    # Set up the SMTP server
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    
    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        # Connect to the server and send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        st.error('Unable to Send Enail due to Server Issue')

def login_page():
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://www.shutterstock.com/image-photo/ai-recruiting-technology-concept-using-600nw-2206168377.jpg');
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
    # Center the login form using Streamlit form layout
    col, col2, col3 = st.columns([2, 4, 2])
    with col2:
        if "otp_sent" not in st.session_state:
            st.session_state["otp_sent"] = False
        if "email" not in st.session_state:
            st.session_state["email"] = None
        if "otp_verified" not in st.session_state:
            st.session_state["otp_verified"] = False
        if not st.session_state["otp_sent"]:
            with st.form(key="login_form"):
                st.markdown('<h1 style="text-align: center; color: red;">Login Here !!!</h1>', unsafe_allow_html=True)
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                col1,col2,col3=st.columns([2,2,1])
                login_button = col2.form_submit_button("Verify Mail", type='primary')

                if login_button:
                    user = validate_user(email, password)
                    if user:
                        otp = random.randint(1000, 9999)
                        to_email=email
                        subject = "OTP for AI Recruiting System"
                        body = f"🤖 Hello,\n\n🔐 Your OTP is {otp}. Please enter this OTP to login.\n\nBest Regards,\n🚀 Team AI Recruitment System"
                        from_email = 'dont.reply.mail.mail@gmail.com'
                        from_password = 'ekdbgizfyaiycmkv'  
                        # Send the alert email
                        send_alert_email(to_email, subject, body, from_email, from_password)
                        update_otp(email, otp)
                        st.session_state["otp_sent"] = True
                        st.session_state["email"] = email
                        st.session_state["user"] = user
                        st.experimental_rerun()  # Rerun script to show OTP form
                    else:
                        st.error("Invalid Email or Password!")

        else:  # OTP Verification Form
            with st.form(key="otp_form"):
                st.markdown('<h1 style="text-align: center; color: red;">Enter OTP</h1>', unsafe_allow_html=True)
                otp_input = st.text_input("Enter OTP",placeholder="Enter OTP that you received on your email")
                col1,col2,col3=st.columns([2,2,1])
                submit_button = col2.form_submit_button("Submit", type='primary')
                if submit_button and otp_input:
                    stored_otp = fetch_otp(st.session_state["email"])[0]
                    if int(otp_input) == int(stored_otp):
                        st.success("Login Successful!")
                        st.session_state["otp_verified"] = True
                        st.session_state["page"] = "user_home"
                        st.experimental_rerun()
                    else:
                        st.error("Invalid OTP! Try again.")

        # Redirect after OTP verification
        if st.session_state.get("otp_verified"):
            st.experimental_rerun()
