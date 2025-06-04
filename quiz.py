import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
import time
import os
import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from speechbrain.pretrained import SepformerSeparation

# -------------------
# Config
# -------------------
AUDIO_SAMPLE_RATE = 16000
MODEL_NAME = "speechbrain/sepformer-wham16k-enhancement"

# -------------------
# Load SpeechBrain Model
# -------------------
@st.cache_resource(show_spinner=False)
def load_speechbrain_model():
    return SepformerSeparation.from_hparams(source=MODEL_NAME, savedir="pretrained_models")

# -------------------
# Text to Speech
# -------------------
def speak(text):
    tts = gTTS(text=text)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    st.audio(fp.read(), format="audio/mp3")
    time.sleep(len(text.split()) * 0.6 + 1)

# -------------------
# Enhance Audio using SpeechBrain
# -------------------
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

# -------------------
# Listen from Microphone
# -------------------
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

# -------------------
# Quiz Logic
# -------------------
questions = [
    {"question": "What is the capital of India?", "answer": "new delhi"},
    {"question": "What is two plus two?", "answer": "4"},
    {"question": "Which planet is known as the red planet?", "answer": "mars"},
    {"question": "What is the color of the sky?", "answer": "blue"},
    {"question": "What is the opposite of hot?", "answer": "cold"},
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

        if q['answer'] in answer:
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

st.set_page_config(page_title="🎤 Smart Audio Quiz", layout="centered")
st.title("🎤 Smart Audio Quiz")

st.checkbox("Enable SpeechBrain Noise Reduction", key="use_speechbrain_checkbox")

if st.button("Start Quiz"):
    run_quiz()
