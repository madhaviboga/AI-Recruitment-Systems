import streamlit as st
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import base64
from Resume_scanner import compare
import seaborn as sns
from matplotlib import pyplot as plt
import pdfplumber
import requests
from bs4 import BeautifulSoup
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
def search_and_display_images(query, num_images=20):
    try:
        # Initialize an empty list for image URLs
        k=[]  
        # Initialize an index for iterating through the list of images
        idx=0  
        # Construct Google Images search URL
        url = f"https://www.google.com/search?q={query}&tbm=isch"  
         # Make an HTTP request to the URL
        response = requests.get(url) 
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")  
        # Initialize an empty list for storing image URLs
        images = []  
        # Iterate through image tags in the HTML content
        for img in soup.find_all("img"):  
             # Limit the number of images to the specified amount
            if len(images) == num_images: 
                break
            # Get the image source URL
            src = img.get("src")  
            # Check if the source URL is valid
            if src.startswith("http") and not src.endswith("gif"):  
                # Add the image URL to the list
                images.append(src)  
        # Iterate through the list of image URLs
        for image in images:  
            # Add each image URL to the list 'k'
            k.append(image)  
        # Reset the index for iterating through the list of image URLs
        idx = 0  
        # Iterate through the list of image URLs
        while idx < len(k):
            # Iterate through the columns in a 4-column layout 
            for _ in range(len(k)): 
                # Create 4 columns for displaying images 
                cols = st.columns(3)  
                # Display the first image in the first column
                cols[0].image(k[idx], width=200)  
                idx += 1 
                # Move to the next image in the list
                cols[1].image(k[idx], width=200)
                # Display the second image in the second column
                idx += 1  
                # Move to the next image in the list
                cols[2].image(k[idx], width=200)  
                # Display the third image in the third column
                idx += 1  
    except:
         # Handle exceptions gracefully if there is an error while displaying images
        pass  

user_data = st.session_state.get('user', None)
def user_profile():
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url('https://media.licdn.com/dms/image/v2/D5610AQF24eAPK2ylIQ/image-shrink_800/image-shrink_800/0/1733809912502?e=2147483647&v=beta&t=xV5UA7sATdv8mGN3K0SuFvhctuQSoXEEF_J9PvvNwdM');
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
        name=user_data[1]
        mail=user_data[2]
        regd_no=user_data[3]
        branch=user_data[4]
        student_type=user_data[5]
        course=user_data[6]
        college_name=user_data[7]
        student_image=user_data[8]
        year_of_study=user_data[9]
        mobile_no=user_data[10]
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
                font-size: 50px;
                font-weight: bold;
                margin-bottom: 15px;
                color: maroon;
            }
            .profile-item {
                font-size: 18px;
                margin-bottom: 10px;
                color: #555;
            }
            .profile-image {
                flex-shrink: 0;
                margin-left: 40px;
                margin-bottom: 250px;
            }
            .profile-image img {
                border-radius: 0%;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.8);
                max-width: 300px;
                max-height: 300px;
            }
            .profile-item {
                margin: 10px 0;
                font-size: 16px;
            }
            .profile-item strong {
                color: red; /* Label color */
            }
            .profile-item {
                color: black; /* Value color */
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
                <div class="profile-item"><strong>Course:</strong> {course}</div>
                <div class="profile-item"><strong>College Name:</strong> {college_name}</div>
                <div class="profile-item"><strong>Email:</strong> {mail}</div>
                <div class="profile-item"><strong>Mobile Number:</strong> {mobile_no}</div>
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
        background-image: url('https://png.pngtree.com/background/20230112/original/pngtree-original-hand-painted-ink-and-watercolor-pure-color-light-colored-paper-picture-image_1999891.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-color: rgba(255, 255, 255, 0.1); /* Add a semi-transparent overlay */
        background-blend-mode: overlay; /* Blend the image with the overlay */
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
    gender = {'Male': 1, 'Female': 0,'Others':0}
    boards_ten = {'Central': 0, 'Others': 1,'State':1}
    boards_twl = {'Central': 0, 'Others': 1,'State':1}
    twl_spl = {'Others': 0, 'HPC': 1, 'MPC': 2,'BIPC':2}
    work_ex = {'Yes': 1, 'No': 0}
    #take inputs in 2 columns
    st.markdown('<div style="display: flex; flex-direction: row;">', unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .stButton button {
        background-color: red; /* Set button color to blue */
        color: white;          /* Set text color to white */
        border-radius: 19px;    /* Optional: Add rounded corners */
        padding: 10px 28px;     /* Optional: Adjust padding */
        border: 2px solid black;          /* Remove border */
        cursor: pointer;       /* Add pointer on hover */
    }
     div.stButton {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


    with st.form(key='my_form'):
        cols = st.columns(2)

        # Take inputs from the user
        g = cols[0].radio(
            "Select your Gender ", ('Male', 'Female','Others'), horizontal=True)
        age = 22
        year_of_study = user_data[9]
        p_ten = cols[1].number_input("Your 10th Percentage", 0, 100, 35)

        b_ten = cols[0].radio(
            "Select your 10th board ", ('State','Central', 'Others'), horizontal=True)

        p_twl = cols[1].number_input("Your 12th/Diploma Percentage", 0, 100, 35)

        b_twl = cols[0].radio(
            "Select your 12th/Diploma Board", ('State','Central', 'Others'), horizontal=True)

        s_twl = cols[0].radio(
            "Select your class 12th stream", ('MPC','BIPC', 'HPC', 'Others'), horizontal=True)

        p_ug = cols[1].number_input("Your UG Degree Percentage",0,100,35)

        s_ug = 1
        p_et=0
        s_pg=0
        wk = cols[0].radio(
            "Do you have previous work experience?", ('Yes', 'No'), horizontal=True)
        p_pg = cols[1].number_input("Your PG Degree Percentage",0,100,35)
        
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
                                boards_twl[b_twl], twl_spl[s_twl], p_ug, s_ug,
                                work_ex[wk], p_et, s_pg, p_pg]])
            if Y_pred[0]:
                st.success("Congrats!! 👏🏻 you are eligible to be placed!!")
                if len(skills) is None:
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
def candidate_matching():
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT5w5dx9JVCCoanCogCi0jdf0zy5PsHLdseKw&s');
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
        <span style='color: red;'>Skill Based Matching</span>
    </h1>
    """,
    unsafe_allow_html=True
    )
    flag = 'HuggingFace-BERT'
    uploaded_files = st.file_uploader(
        '**Upload your Resume File:** ', type="pdf", accept_multiple_files=True)
    JD = st.text_area("**Enter Job Description:**")
    col1,col2,col3 = st.columns([2,1,2])
    comp_pressed = col2.button("Get Score",type='primary')
    if comp_pressed and uploaded_files:
        # Streamlit file_uploader gives file-like objects, not paths
        uploaded_file_paths = [extract_pdf_data(
            file) for file in uploaded_files]
        score = compare(uploaded_file_paths, JD, flag)
    my_dict = {}
    if comp_pressed and uploaded_files:
        for i in range(len(score)):
            # Populate the dictionary with file names and corresponding scores
            my_dict[uploaded_files[i].name] = score[i]
        
        # Sort the dictionary by keys
        sorted_dict = dict(sorted(my_dict.items()))
        
        # Convert the sorted dictionary to a list of tuples
        ct_items = list(sorted_dict.items())
        score = ct_items[0][1]  # Access the score of the first file
        sc=float(score)
        if sc >= 75:
            #print scrore
            sc=round(sc,2)
            st.markdown(f'<p style="text-align: center; font-size: 30px;"><span style="color: green;">Score {sc}%.</span></h1>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <h1 style='text-align: center; font-size: 50px;'>
                    <span style='color: green;'>The Candiate is good match for the Job.</span>
                </h1>
                """,
                unsafe_allow_html=True
            )
            st.image("https://media.istockphoto.com/id/1385218939/vector/people-crowd-applause-hands-clapping-business-teamwork-cheering-ovation-delight-vector.jpg?s=612x612&w=0&k=20&c=7NMaUB4zGoXoePxiy-XxKap53GMBQvmIYOSW1tVSFMY=", use_column_width=True)
            st.balloons()
        elif sc >= 50:
            st.markdown(f'<p style="text-align: center; font-size: 30px;"><span style="color: green;">Score {sc}%.</span></h1>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <h1 style='text-align: center; font-size: 50px;'>
                    <span style='color: orange;'>The Candiate is moderate match for the Job.</span>
                </h1>
                """,
                unsafe_allow_html=True
            )
            # place horizontal line
            st.markdown(
                """
                <hr style='border: 1px solid green;'>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <p style='text-align: center; font-size: 30px; font-family: Garamond, sans-serif;'>
                    <span style='color: purple;'>Improvements</span>
                </p>
                """,
                unsafe_allow_html=True
            )
            #parse Job Description and Extract the skills
            JD = preprocess_text(JD)
            scored_df = calculate_scores(JD)
            high_scores_df = scored_df[scored_df['Score'] > 1]
            if not high_scores_df.empty:
                #make a list of skills
                data = pd.read_csv('image.csv')
                # Number of columns per row
                columns_per_row = 3
                # Count the total number of rows needed
                total_items = len(high_scores_df)
                total_rows = -(-total_items // columns_per_row)  # Ceiling division

                for row_index in range(total_rows):
                    # Create a new row of columns
                    cols = st.columns(columns_per_row)
                    
                    # Iterate through the items in this row
                    for col_index in range(columns_per_row):
                        # Calculate the actual index in the dataframe
                        item_index = row_index * columns_per_row + col_index
                        
                        # Check if the index is valid
                        if item_index < total_items:
                            # Get the row data
                            row = high_scores_df.iloc[item_index]
                            skill_name = row['Domain/Area']
                            img = data[data['Skill'] == str(skill_name)]['Image'].values[0]
                            cols[col_index].write(skill_name)
                            cols[col_index].image(img, width=150)
 
            else:
                col1,col2,col3 = st.columns([1,2,1])
                col2.error("No Improvements Needed...")    
        else:
            st.markdown(f'<p style="text-align: center; font-size: 30px;"><span style="color: green;">Score {sc}%.</span></h1>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <h1 style='text-align: center; font-size: 50px;'>
                    <span style='color: red;'>The Candiate is not a good match for the Job.</span>
                </h1>
                """,
                unsafe_allow_html=True
            )
            # place horizontal line
            st.markdown(
                """
                <hr style='border: 1px solid green;'>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <p style='text-align: center; font-size: 30px; font-family: Garamond, sans-serif;'>
                    <span style='color: purple;'>Improvements</span>
                </p>
                """,
                unsafe_allow_html=True
            )
            #parse Job Description and Extract the skills
            JD = preprocess_text(JD)
            scored_df = calculate_scores(JD)
            high_scores_df = scored_df[scored_df['Score'] >= 1]
            if not high_scores_df.empty:
                #make a list of skills
                data = pd.read_csv('image.csv')
                # Number of columns per row
                columns_per_row = 3
                # Count the total number of rows needed
                total_items = len(high_scores_df)
                total_rows = -(-total_items // columns_per_row)  # Ceiling division

                for row_index in range(total_rows):
                    # Create a new row of columns
                    cols = st.columns(columns_per_row)
                    
                    # Iterate through the items in this row
                    for col_index in range(columns_per_row):
                        # Calculate the actual index in the dataframe
                        item_index = row_index * columns_per_row + col_index
                        
                        # Check if the index is valid
                        if item_index < total_items:
                            # Get the row data
                            row = high_scores_df.iloc[item_index]
                            skill_name = row['Domain/Area']
                            img = data[data['Skill'] == str(skill_name)]['Image'].values[0]
                            
                            # Display the skill name and image
                            cols[col_index].write(skill_name)
                            cols[col_index].image(img, width=150)
            else:
                col1,col2,col3 = st.columns([1,2,1])
                col2.error("No Improvements Needed...") 
def interview():
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url('https://plus.unsplash.com/premium_photo-1672423154405-5fd922c11af2?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Y29ycG9yYXRlJTIwYmFja2dyb3VuZHxlbnwwfHwwfHx8MA%3D%3D');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-color: rgba(255, 255, 255, 0.7); /* Add a semi-transparent overlay */
        background-blend-mode: overlay; /* Blend the image with the overlay */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: green;'>Recruitment Process</span>
    </h1>
    """,
    unsafe_allow_html=True
    )
    data=pd.read_csv('company.csv',encoding='latin1')
    company_names=data['Company'].tolist()
    #display the company names in alphabetical order
    company_names.sort()
    company_name = st.selectbox("Select Company Name",company_names)
    search_and_display_images(company_name+'comapany',3)
    # draw horizontal line
    st.markdown(
        """
        <hr style='border: 1px solid green;'>
        """,
        unsafe_allow_html=True
    )
    try:
        #About the company which is in Info column
        info=data[data['Company']==company_name]['Info'].values[0]
        st.write(info)
        round1 = data[data['Company'] == company_name]['Round 1'].values[0]
        round2 = data[data['Company'] == company_name]['Round 2'].values[0]
        round3 = data[data['Company'] == company_name]['Round 3'].values[0]
        round4 = data[data['Company'] == company_name]['Round 4'].values[0]
        round5 = data[data['Company'] == company_name]['Round 5'].values[0]
        round6 = data[data['Company'] == company_name]['Round 6'].values[0]
        # Helper function to style headings and text
        def style_round(round_heading, round_text):
            # Separate heading from details using ":"
            heading, details = round_text.split(":", 1)
            # Style the heading and details
            styled_heading = f"<span style='font-weight:bold;'>{heading}:</span>"
            styled_details = f"<span>{details.strip()}</span>"
            return f"<p style='color:red; font-weight:bold;'>{round_heading}</p><p>{styled_heading} {styled_details}</p>"

        # Display rounds with styled text
        st.markdown(style_round("Round 1", round1), unsafe_allow_html=True)
        st.markdown(style_round("Round 2", round2), unsafe_allow_html=True)
        st.markdown(style_round("Round 3", round3), unsafe_allow_html=True)
        st.markdown(style_round("Round 4", round4), unsafe_allow_html=True)
        st.markdown(style_round("Round 5", round5), unsafe_allow_html=True)
        st.markdown(style_round("Round 6", round6), unsafe_allow_html=True)
    except:
        pass
def user_home_page():
    # Navigation menu for user dashboard
    if "show_logout_modal" not in st.session_state:
        st.session_state.show_logout_modal = False
    with st.sidebar:
        selected_tab = option_menu(
            menu_title=None,
            options=["Student Profile", "Placement Prediction", "Resume Screening",'Candidate Matching','Interview Process','Logout'],
            icons=['person-circle','bar-chart','newspaper','person-check','info-circle-fill','unlock-fill'], menu_icon="cast", default_index=0,
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
    elif selected_tab == "Candidate Matching":
        candidate_matching()
    elif selected_tab == "Interview Process":
        interview()
    elif selected_tab=='Logout':
        st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://st2.depositphotos.com/3252397/7026/i/450/depositphotos_70265521-stock-photo-keyboard-logout-black.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.8); /* Add a semi-transparent overlay */
            background-blend-mode: overlay; /* Blend the image with the overlay */
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        st.session_state.show_logout_modal = True   
        if st.session_state.show_logout_modal:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            col1,col2,col3=st.columns([1,4,1])
            with col2.form(key="logout"):
                st.error("Are you sure you want to log out?")
                col1, col2, col3 = st.columns([1, 2, 1])

                with col1:
                    if st.form_submit_button("Yes, Logout",type='primary'):
                        st.session_state.clear()
                        st.experimental_rerun()

                with col3:
                    if st.form_submit_button("Cancel",type='primary'):
                        st.session_state.show_logout_modal = False  # Close dialog
