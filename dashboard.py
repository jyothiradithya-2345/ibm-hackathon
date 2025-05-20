import streamlit as st
import pandas as pd
from pymongo import MongoClient
import streamlit.components.v1 as components
import speech_recognition as sr
from gtts import gTTS
import io
import time
import numpy as np
import soundfile as sf
from speechbrain.pretrained import SepformerSeparation

# --- Page config ---
st.set_page_config(page_title="Learning Dashboard", layout="wide")

# --- MongoDB Setup ---
@st.cache_resource
def get_mongo_client():
    return MongoClient("mongodb://localhost:27017")

client = get_mongo_client()
db = client["learning_dashboard"]
courses_col = db["courses"]
recs_col = db["recommendations"]

# --- Initialize Sample Data if Empty ---
if courses_col.count_documents({}) == 0:
    courses_col.insert_many([
        {"Course": "Data Structures in Python", "Progress": "60%", "Status": "In Progress"},
        {"Course": "Machine Learning Basics", "Progress": "80%", "Status": "In Progress"},
        {"Course": "Web Development with Django", "Progress": "100%", "Status": "Completed"},
    ])
if recs_col.count_documents({}) == 0:
    recs_col.insert_many([
        {"text": "Try the new 'Advanced SQL Queries' course."},
        {"text": "Revise 'OOP in Python' for better practice."},
        {"text": "Explore 'Deep Learning with PyTorch' as a next step."},
    ])

# --- Utility Functions ---
def load_courses():
    return list(courses_col.find({}, {"_id": 0}))

def load_recommendations():
    return [r["text"] for r in recs_col.find({}, {"_id": 0})]

def insert_course(course, progress, status):
    courses_col.insert_one({"Course": course, "Progress": progress, "Status": status})

def insert_recommendation(text):
    recs_col.insert_one({"text": text})

# --- Pages ---
def page_home():
    st.title("📚 Home - Learning Dashboard")

    courses = load_courses()
    recs = load_recommendations()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Ongoing Courses")
        if courses:
            df = pd.DataFrame(courses)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No courses found.")

    with col2:
        st.subheader("📈 Progress Summary")
        if courses:
            progress_data = {
                "Course": [c["Course"] for c in courses],
                "Completion (%)": [int(c["Progress"].replace('%', '')) for c in courses]
            }
            df = pd.DataFrame(progress_data)
            st.bar_chart(df.set_index("Course"))
        else:
            st.info("No progress data.")

    st.markdown("---")
    st.subheader("🎯 Recommendations")
    if recs:
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.info("No recommendations available.")

def page_my_courses():
    st.title("📘 My Courses")

    courses = load_courses()
    st.subheader("Current Courses")
    if courses:
        df = pd.DataFrame(courses)
        st.table(df)
    else:
        st.info("No courses found.")

    st.subheader("➕ Add New Course")
    with st.form("add_course"):
        name = st.text_input("Course Name")
        progress = st.slider("Progress (%)", 0, 100, 0)
        status = st.selectbox("Status", ["In Progress", "Completed"])
        submitted = st.form_submit_button("Add Course")
        if submitted and name:
            insert_course(name, f"{progress}%", status)
            st.success(f"Course '{name}' added successfully!")

def page_add_recommendation():
    st.title("🧐 Add Recommendation")

    with st.form("add_rec"):
        rec_text = st.text_area("Enter recommendation")
        submitted = st.form_submit_button("Add")
        if submitted and rec_text:
            insert_recommendation(rec_text)
            st.success("Recommendation added!")

    st.subheader("📌 Current Recommendations")
    recs = load_recommendations()
    if recs:
        st.markdown("\n".join([f"> {r}" for r in recs]))
    else:
        st.info("No recommendations found.")

def page_settings():
    st.title("⚙️ Settings")
    st.subheader("Theme Preferences")
    theme = st.radio("Choose theme:", ["Light", "Dark"], index=0)

    if theme == "Dark":
        components.html("""
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        </style>
        """, height=0)
    else:
        components.html("""
        <style>
        body {
            background-color: white;
            color: black;
        }
        .stApp {
            background-color: white;
            color: black;
        }
        </style>
        """, height=0)

    st.subheader("User Preferences")
    username = st.text_input("Enter your display name")
    notifications = st.checkbox("Enable notifications")
    st.success("Preferences saved locally (not stored in DB yet).")

# --- Smart Audio Quiz Setup ---
AUDIO_SAMPLE_RATE = 16000
MODEL_NAME = "speechbrain/sepformer-wham16k-enhancement"

@st.cache_resource(show_spinner=False)
def load_speechbrain_model():
    return SepformerSeparation.from_hparams(source=MODEL_NAME, savedir="pretrained_models")

def speak(text):
    tts = gTTS(text=text)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    st.audio(fp.read(), format="audio/mp3")
    time.sleep(len(text.split()) * 0.6 + 1)

def enhance_audio(input_audio, sample_rate=16000):
    raw_data = input_audio.get_raw_data(convert_rate=sample_rate, convert_width=2)
    np_audio = np.frombuffer(raw_data, np.int16).astype(np.float32) / 32768.0
    sf.write("input.wav", np_audio, sample_rate)
    
    separator = load_speechbrain_model()
    est_sources = separator.separate_file(path="input.wav")
    enhanced = est_sources[0].squeeze().cpu().numpy()
    
    sf.write("enhanced.wav", enhanced, sample_rate)
    with open("enhanced.wav", "rb") as f:
        enhanced_bytes = f.read()
    
    return sr.AudioData(enhanced_bytes, sample_rate, 2)

def listen():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 1.0

    try:
        with sr.Microphone(sample_rate=AUDIO_SAMPLE_RATE) as source:
            st.info("Listening…")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=7)

        if st.session_state.get("use_speechbrain_checkbox", False):
            st.info("Enhancing audio...")
            audio = enhance_audio(audio)

        response = recognizer.recognize_google(audio)
        st.success(f"You said: {response}")
        return response.lower()
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
    except sr.RequestError:
        st.error("Could not connect to recognition service.")
    except Exception as e:
        st.error(f"Microphone error: {e}")
    return ""

def listen_with_retry(retries=2):
    for _ in range(retries):
        result = listen()
        if result:
            return result
        speak("I did not catch that. Please say it again.")
    return ""

# Quiz questions related to multiple CS subjects including DBMS, Java, etc.
questions = [
    {"question": "What does SQL stand for in databases?", "answer": "structured query language"},
    {"question": "Name the four main concepts of Object Oriented Programming in Java.", 
     "answer": "encapsulation inheritance polymorphism abstraction"},
    {"question": "Which command is used to remove all rows from a table in SQL?", "answer": "truncate"},
    {"question": "What is a primary key in a database?", 
     "answer": "a unique identifier for each record"},
    {"question": "Which keyword is used to create a new object in Java?", "answer": "new"},
]

def run_quiz():
    score = 0
    speak("Welcome to the Smart Audio Quiz.")
    st.write("Welcome to the Smart Audio Quiz.")

    speak("What is your name?")
    name = listen_with_retry().title()
    if not name:
        name = "Student"

    speak(f"Hello {name}, let's begin the quiz.")
    st.write(f"Hello {name}, let's begin the quiz.")

    for i, q in enumerate(questions, start=1):
        speak(f"Question {i}: {q['question']}")
        st.write(f"Q{i}: {q['question']}")
        speak("Your answer?")
        answer = listen_with_retry()

        if all(word in answer for word in q['answer'].split()):
            speak("Correct.")
            st.success("Correct!")
            score += 1
        else:
            speak(f"Incorrect. The correct answer is {q['answer']}.")
            st.error(f"Incorrect. Correct: {q['answer']}")

        time.sleep(1.5)

    speak(f"{name}, your score is {score} out of {len(questions)}.")
    st.success(f"Final Score: {score}/{len(questions)}")
    speak("Thanks for playing.")

# --- Student Activity Page ---
def page_student_activity():
    st.title("🎤 Student Activity - Smart Audio Quiz")
    st.checkbox("Enable SpeechBrain Noise Reduction", key="use_speechbrain_checkbox")

    if st.button("Start Quiz"):
        run_quiz()

# --- Main App ---
def main():
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to", ["Home", "My Courses", "Add Recommendation", "Student Activity", "Settings"])

    if page == "Home":
        page_home()
    elif page == "My Courses":
        page_my_courses()
    elif page == "Add Recommendation":
        page_add_recommendation()
    elif page == "Student Activity":
        page_student_activity()
    elif page == "Settings":
        page_settings()

if __name__ == "__main__":
    main()
