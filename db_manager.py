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
            mobile_no VARCHAR(15),
            password TEXT

        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS otp_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            otp INTEGER
        )
    """)
    conn.commit()
    conn.close()

# Register a new user
def register_user(name, email, regd_no, branch, student_type, course, college_name, student_image, year_of_study, mobile_no, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        # Read the image file as a binary stream
        if student_image is not None:
            student_image = student_image.read()

        cursor.execute("""
            INSERT INTO users (name, email, regd_no, branch, student_type, course, college_name, student_image, year_of_study, mobile_no, password) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?)
        """, (name, email, regd_no, branch, student_type, course, college_name, student_image, year_of_study, mobile_no, password))
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