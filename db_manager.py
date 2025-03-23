import sqlite3

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    #drop table
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            regd_no VARCHAR(20),
            branch TEXT,
            student_type TEXT,
            course TEXT,
            college_name TEXT,
            student_image BLOB,
            year_of_study INTEGER,
            mobile_no VARCHAR(10),
            otp INTEGER,
            password TEXT

        )
    """)
    conn.commit()
    conn.close()

# Register a new user
def register_user(name, email, regd_no, branch, student_type, course, college_name, student_image, year_of_study, mobile_no, otp,password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        # Read the image file as a binary stream
        if student_image is not None:
            student_image = student_image.read()

        cursor.execute("""
            INSERT INTO users (name, email, regd_no, branch, student_type, course, college_name, student_image, year_of_study, mobile_no,otp, password) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?,?)
        """, (name, email, regd_no, branch, student_type, course, college_name, student_image, year_of_study, mobile_no,otp, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Validate login credentials
def validate_user(email, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()
    conn.close()
    return user


def valid_user(email):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def fetch_otp(email):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT otp FROM users WHERE email = ?", (email,))
    otp = cursor.fetchone()
    conn.close()
    return otp

def update_otp(email,otp):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET otp = ? WHERE email = ?", (otp,email))
    conn.commit()
    conn.close()

def change_password(email,password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password = ? WHERE email = ?", (password,email))
    conn.commit()
    conn.close()
def fetch_password(email):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
    password = cursor.fetchone()
    conn.close()
    return password

def edit_profile(name, email, regd_no, year_of_study, mobile_no):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET name = ?, regd_no = ?, year_of_study = ?, mobile_no = ? WHERE email = ?", (name, regd_no, year_of_study, mobile_no, email))
    conn.commit()
    conn.close()

def fetch_user(email):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user