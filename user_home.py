import streamlit as st
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import base64
import seaborn as sns
from matplotlib import pyplot as plt
import pdfplumber
import base64
import PyPDF2
import re
import string
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = reader.numPages
    content = ""
    for page_number in range(number_of_pages):
        page = reader.getPage(page_number)
        content += page.extractText()
    return content
#load the model
# Function to preprocess text
def preprocess_text(content):
    content = content.lower()
    content = re.sub(r'[0-9]+', '', content)
    content = content.translate(str.maketrans('', '', string.punctuation))
    return content

def extract_pdf_data(file_path):
    data = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data += text
    return data


def extract_text_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data
# Function to calculate scores
def calculate_scores(content):
    Area_with_key_term = {
        'Data science': ['algorithm', 'analytics', 'hadoop', 'machine learning', 'data mining', 'python',
                        'statistics', 'data', 'statistical analysis', 'data wrangling', 'algebra', 'Probability',
                        'visualization'],
        'Programming': ['python', 'r programming', 'sql', 'c++', 'scala', 'julia', 'tableau', 'javascript',
                        'powerbi', 'code', 'coding'],
        'Experience': ['project', 'years', 'company', 'excellency', 'promotion', 'award', 'outsourcing', 'work in progress'],
        'Management skill': ['administration', 'budget', 'cost', 'direction', 'feasibility analysis', 'finance', 
                            'leader', 'leadership', 'management', 'milestones', 'planning', 'problem', 'project', 
                            'risk', 'schedule', 'stakeholders', 'English'],
        'Data analytics': ['api', 'big data', 'clustering', 'code', 'coding', 'data', 'database', 'data mining', 
                        'data science', 'deep learning', 'hadoop', 'hypothesis test', 'machine learning', 'dbms', 
                        'modeling', 'nlp', 'predictive', 'text mining', 'visualization'],
        'Statistics': ['parameter', 'variable', 'ordinal', 'ratio', 'nominal', 'interval', 'descriptive', 
                        'inferential', 'linear', 'correlations', 'probability', 'regression', 'mean', 'variance', 
                        'standard deviation'],
        'Machine learning': ['supervised learning', 'unsupervised learning', 'ann', 'artificial neural network', 
                            'overfitting', 'computer vision', 'natural language processing', 'database'],
        'Data analyst': ['data collection', 'data cleaning', 'data processing', 'interpreting data', 
                        'streamlining data', 'visualizing data', 'statistics', 'tableau', 'tables', 'analytical'],
        'Software': ['django', 'cloud', 'gcp', 'aws', 'javascript', 'react', 'redux', 'es6', 'node.js', 
                    'typescript', 'html', 'css', 'ui', 'ci/cd', 'cashflow'],
        'Web skill': ['web design', 'branding', 'graphic design', 'seo', 'marketing', 'logo design', 'video editing', 
                    'es6', 'node.js', 'typescript', 'html/css', 'ci/cd'],
        'Personal Skill': ['leadership', 'team work', 'integrity', 'public speaking', 'team leadership', 
                            'problem solving', 'loyalty', 'quality', 'performance improvement', 'six sigma', 
                            'quality circles', 'quality tools', 'process improvement', 'capability analysis', 
                            'control'],
        'Accounting': ['communication', 'sales', 'sales process', 'solution selling', 'crm', 'sales management', 
                    'sales operations', 'marketing', 'direct sales', 'trends', 'b2b', 'marketing strategy', 
                    'saas', 'business development'],
        'Sales & marketing': ['retail', 'manufacture', 'corporate', 'goods sale', 'consumer', 'package', 'fmcg', 
                            'account', 'management', 'lead generation', 'cold calling', 'customer service', 
                            'inside sales', 'sales', 'promotion'],
        'Graphic': ['brand identity', 'editorial design', 'design', 'branding', 'logo design', 'letterhead design', 
                    'business card design', 'brand strategy', 'stationery design', 'graphic design', 'exhibition graphic design'],
        'Content skill': ['editing', 'creativity', 'content idea', 'problem solving', 'writer', 'content thinker', 
                        'copy editor', 'researchers', 'technology geek', 'public speaking', 'online marketing'],
        'Graphical content': ['photographer', 'videographer', 'graphic artist', 'copywriter', 'search engine optimization', 
                            'seo', 'social media', 'page insight', 'gain audience'],
        'Finanace': ['financial reporting', 'budgeting', 'forecasting', 'strong analytical thinking', 'financial planning', 
                    'payroll tax', 'accounting', 'productivity', 'reporting costs', 'balance sheet', 'financial statements'],
        'Health/Medical': ['abdominal surgery', 'laparoscopy', 'trauma surgery', 'adult intensive care', 'pain management', 
                        'cardiology', 'patient', 'surgery', 'hospital', 'healthcare', 'doctor', 'medicine'],
        'Language': ['english', 'malay', 'mandarin', 'bangla', 'hindi', 'tamil']
    }

    scores = {domain: sum(1 for word in terms if word in content) for domain, terms in Area_with_key_term.items()}
    return pd.DataFrame(list(scores.items()), columns=['Domain/Area', 'Score']).sort_values(by='Score', ascending=False)
user_data = st.session_state.get('user', None)
def user_profile():
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url('https://img.freepik.com/free-vector/geometric-science-education-background-vector-gradient-blue-digital-remix_53876-125993.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    # Extracting user data from session state after successful login
    user_data = st.session_state.get('user', None)
    
    if user_data:
        # Assuming 'user' is a tuple (id, name, email, password, regd_no, year_of_study, branch, student_type, student_image)
        name, regd_no, year_of_study, branch, student_type, student_image = user_data[1], user_data[4], user_data[5], user_data[6], user_data[7], user_data[8]

        # Check if the student_image is a bytes-like object
        if isinstance(student_image, bytes):
            # Encode the binary image to base64
            image_data = base64.b64encode(student_image).decode()
            image_link = f"data:image/png;base64,{image_data}"
        elif isinstance(student_image, str) and student_image:  # In case it's a file path or URL
            try:
                # Open the image as binary if it's a valid file path
                with open(student_image, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode()
                    image_link = f"data:image/png;base64,{image_data}"
            except FileNotFoundError:
                # Default image in case file is not found
                image_link = "https://cdn-icons-png.flaticon.com/512/4042/4042356.png"
        else:
            # Default image if no image data is available
            image_link = "https://cdn-icons-png.flaticon.com/512/4042/4042356.png"

        # CSS Styling
        profile_css = """
        <style>
            .profile-container {
                background-color: #f7e7e6;
                padding: 60px;
                border-radius: 50px;
                box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
                max-width: 1000px;
                margin: auto;
                font-family: Arial, sans-serif;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .profile-details {
                flex: 1;
            }
            .profile-header {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }
            .profile-item {
                font-size: 18px;
                margin-bottom: 10px;
                color: #555;
            }
            .profile-image {
                flex-shrink: 0;
                margin-left: 40px;
            }
            .profile-image img {
                border-radius: 50%;
                max-width: 300px;
                max-height: 300px;
            }
        </style>
        """

        # HTML Structure
        profile_html = f"""
        <div class="profile-container">
            <div class="profile-details">
                <div class="profile-header">Student Report</div>
                <div class="profile-item"><strong>Name:</strong> {name}</div>
                <div class="profile-item"><strong>Registration Number:</strong> {regd_no}</div>
                <div class="profile-item"><strong>Year of Study:</strong> {year_of_study}</div>
                <div class="profile-item"><strong>Branch:</strong> {branch}</div>
                <div class="profile-item"><strong>Student Type:</strong> {student_type}</div>
            </div>
            <div class="profile-image">
                <img src="{image_link}" alt="User Image">
            </div>
        </div>
        """

        # Display styled content
        st.markdown(profile_css + profile_html, unsafe_allow_html=True)
    else:
        st.error("User not logged in!")


def placement():
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url('https://t3.ftcdn.net/jpg/05/00/34/58/360_F_500345899_4OqmtspFst6SRnNQvLj7h7TfKOrBwTer.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: #7018b8;'>Student Placement Prediction</span>
    </h1>
    """,
    unsafe_allow_html=True
    )
    df = pd.read_csv("clean_data.csv")
    sc = [0]

    def model():
        X = df.iloc[:, :-1].values # features
        Y = df.iloc[:, -1].values  # target

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                max_iter=1000).fit(X_train, Y_train)

        sc[0] = clf.score(X_test, Y_test)
        return clf

    sta = 0
    gender = {'M': 1, 'F': 0}
    boards_ten = {'Central': 0, 'Others': 1}
    boards_twl = {'Central': 0, 'Others': 1}
    twl_spl = {'Others': 0, 'Comm': 1, 'Sci': 2}
    ug_spl = {'Others': 1, 'Comm': 0, 'Sci': 2}
    work_ex = {'Yes': 1, 'No': 0}
    pg_spl = {"Mkt/hr": 1, "Mkt/fin": 0}
    #take inputs in 2 columns
    st.markdown('<div style="display: flex; flex-direction: row;">', unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .stButton button {
        background-color: blue; /* Set button color to blue */
        color: white;          /* Set text color to white */
        border-radius: 5px;    /* Optional: Add rounded corners */
        padding: 8px 16px;     /* Optional: Adjust padding */
        border: none;          /* Remove border */
        cursor: pointer;       /* Add pointer on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)


    with st.form(key='my_form'):
        cols = st.columns(2)

        # Take inputs from the user
        g = cols[0].radio(
            "Select your Gender ", ('M', 'F'), horizontal=True)
        age = 20
        year_of_study = 2  # Example year of study
        p_ten = cols[1].slider("Your 10th Boards Percentage", 0, 100)

        b_ten = cols[0].radio(
            "Select your 10th board ", ('Central', 'Others'), horizontal=True)

        p_twl = cols[1].slider("Your 12th Boards Percentage", 0, 100)

        b_twl = cols[0].radio(
            "Select your 12th board ", ('Central', 'Others'), horizontal=True)

        s_twl = cols[0].radio(
            "Select your class 12th stream", ('Sci', 'Comm', 'Others'), horizontal=True)

        p_ug = cols[1].slider("Your UG Degree Percentage", 0, 100)

        s_ug = cols[0].radio(
            "Select your UG Degree ", ('Sci', 'Comm', 'Others'), horizontal=True)

        wk = cols[0].radio(
            "Do you have previous work experience?", ('Yes', 'No'), horizontal=True)

        p_et = cols[1].slider("Your Percentage in Employability test", 0, 100)

        s_pg = cols[0].radio(
            "Select your MBA specialization ", ("Mkt/hr", "Mkt/fin"), horizontal=True)

        p_pg = cols[1].slider("Your PG Degree Percentage", 0, 100)
        
        skills = st.multiselect("Select your skills", [
            'Python', 'C', 'Java', 'C++', 'C#', 'R', 'Javascript', 'Data Base', 'SQL', 'NOSQL',
            'RDBMS', 'Data Structures', 'Data Science', 'Data Mining', 'Data Analysis', 
            'Machine Learning', 'NLP', 'Deep Learning', 'AI', 'GenAI', 'HTML', 'CSS', 
            'Angular', 'React', 'Node', 'Express', 'MongoDB', 'Big Data', 'Hadoop', 
            'Cloud Computing', 'Azure', 'Operating Systems', 'Software Engineering', 
            'Computer Networks'
        ])

        # Add the submit button with a div for centering
        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        sta = 1
        if sta:
            x = model()
            Y_pred = x.predict([[gender[g], p_ten, boards_ten[b_ten], p_twl,
                                boards_twl[b_twl], twl_spl[s_twl], p_ug, ug_spl[s_ug],
                                work_ex[wk], p_et, pg_spl[s_pg], p_pg]])
            if Y_pred[0]:
                st.success("Congrats!! 👏🏻 you are eligible to be placed!!")
                if len(skills)<1:
                    st.write("<p style='color: red;'>You need to improve your skills!</p>", unsafe_allow_html=True)          

            else:
                st.error("Sorry!! you are not eligible to be placed!!")
                #needs to improve the skills based on the scores
                

def resume():
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url('https://img.freepik.com/free-vector/white-abstract-background_23-2148806276.jpg?semt=ais_hybrid');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """ 
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: indigo;'>Resume Screening</span>
    </h1>
    """,
    unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload a Resume (PDF)", type="pdf")

    if uploaded_file:
        # Extract text
        # Extract and preprocess the text
        content = extract_text_from_pdf(uploaded_file)
        content = preprocess_text(content)

        # Calculate scores
        scored_df = calculate_scores(content)

        # Filter high scores (e.g., scores greater than 80)
        high_scores_df = scored_df[scored_df['Score'] > 1]

        # Display high scores
        st.markdown(
            """
            <p style='font-size: 30px;'>
                <span style='color: red;'>Resume</span>
                <span style='color: red;'>Scores</span>
            </p>
            """,
            unsafe_allow_html=True
        )
        if not high_scores_df.empty:
            # display in 3 cols, each col has skill and score in percentage
            cols = st.columns(3)
            for i, row in high_scores_df.iterrows():
                cols[i % 3].write(f"{row['Domain/Area']}")
                
                # Check if the Score is a fraction (0 to 1) or a percentage (0 to 100)
                if row['Score'] <= 1:
                    # Score is a fraction, so multiply by 100 to get percentage
                    score_percentage = row['Score'] * 100
                else:
                    # Score is already a percentage
                    score_percentage = row['Score']  # Do not multiply by sum(scored_df['Score'])
                
                # Clamp the score between 0 and 100
                score_percentage = max(0, min(100, score_percentage))

                # Display progress bar with the score
                cols[i % 3].progress(score_percentage)

            # Filter out rows where the score is 0
            filtered_df = scored_df[scored_df['Score'] > 0]

            # Visualization: Bar chart for scores by domain
            st.markdown(
                """
                <p style='font-size: 30px;'>
                    <span style='color: indigo;'>Domain</span>
                    <span style='color: indigo;'>Expertise</span>
                </p>
                """,
                unsafe_allow_html=True
            )

            # Set the figure size
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create a bar chart using seaborn to get a nice color palette
            sns.barplot(x='Domain/Area', y='Score', data=filtered_df, ax=ax, palette='viridis')

            # Rotate the x-axis labels for better visibility
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            # Set labels and title
            ax.set_xlabel('Domain')
            ax.set_ylabel('Score')

            # Display the plot
            st.pyplot(fig)


            # Analyze and recommend
            total_score = scored_df['Score'].sum()
            st.markdown(
                """
                <p style='font-size: 30px;'>
                    <span style='color: green;'>Application Status</span>
                </p>
                """,
                unsafe_allow_html=True
            )
            # if data scientist has high score in data science, programming, statistics, machine learning, data analytics
            if total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Data science', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Programming', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Statistics', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Machine learning', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Data analytics', 'Score'].values[0] >= 1:
                st.success("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Data Scientist.")
            # if account executive has high score in sales & marketing, personal skill, management skill, experience
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Sales & marketing', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Personal Skill', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Management skill', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Experience', 'Score'].values[0] >= 1:
                st.success("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Account Executive.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Content skill', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Graphic', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Personal Skill', 'Score'].values[0] >= 1:
                st.success("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Content Creator.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Data analytics', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Statistics', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Data analyst', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Data science', 'Score'].values[0] >= 1:
                st.success("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Data Analyst.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Sales & marketing', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Accounting', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Management skill', 'Score'].values[0] >= 1 :
                st.success("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Sales Executive.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Programming', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Software', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Experience', 'Score'].values[0] >= 1:
                st.success("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Software Engineer.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Web skill', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Graphic', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Personal Skill', 'Score'].values[0] >= 1:
                st.success("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Web and Graphic Designer.")
            else:
                st.error("Resume Does Not Meet The Requirement for any Role Based on Scores.")
    
        else:
            st.write("Resume Does Not Meet The Requirement for any Role.")
def user_home_page():
    # Navigation menu for user dashboard
    with st.sidebar:
        selected_tab = option_menu(
            menu_title=None,
            options=["Student Profile", "Placement Prediction", "Resume Screening",'Logout'],
            icons=['person-circle','bar-chart','newspaper','unlock-fill'], menu_icon="cast", default_index=0,
        styles={
        "nav-link-selected": {"background-color": "skyblue", "color": "black", "border-radius": "5px"},
        }
        )
    if selected_tab == "Student Profile":
        user_profile()
    elif selected_tab == "Placement Prediction":
        placement()
    elif selected_tab == "Resume Screening":
        resume()
    elif selected_tab=='Logout':
        # Logout functionality
        st.session_state.clear()  # Clear session state to "log out"
        st.experimental_rerun()