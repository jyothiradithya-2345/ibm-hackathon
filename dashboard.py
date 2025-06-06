# streamlit_app.py
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import speech_recognition as sr
from gtts import gTTS
import io
import time
import numpy as np
import soundfile as sf
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModel # Import AutoModel for embeddings
from huggingface_hub import login
from speechbrain.pretrained import SepformerSeparation
import bcrypt # For password hashing
from sklearn.metrics.pairwise import cosine_similarity # For similarity search

# --- Hugging Face Login ---
login("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")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Learning Dashboard", layout="wide")

# --- MongoDB ---
@st.cache_resource
def get_mongo_client():
    return MongoClient("mongodb://localhost:27017")

client = get_mongo_client()
db = client["learning_dashboard"]
courses_col = db["courses"]
recs_col = db["recommendations"]
users_col = db["users"] # Collection for users

# --- Initialize Sample Data (Courses and Recommendations) ---
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

# --- TTS Utility ---
def speak(text):
    tts = gTTS(text=text)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    st.audio(fp.read(), format="audio/mp3")
    time.sleep(len(text.split()) * 0.5 + 0.8)

# --- Speech Enhancement ---
@st.cache_resource
def load_speechbrain_model():
    return SepformerSeparation.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir="pretrained_models")

def enhance_audio(input_audio, sample_rate=16000):
    raw_data = input_audio.get_raw_data(convert_rate=sample_rate, convert_width=2)
    np_audio = np.frombuffer(raw_data, np.int16).astype(np.float32) / 32768.0

    input_wav_path = "input.wav"
    enhanced_wav_path = "enhanced.wav"
    sf.write(input_wav_path, np_audio, sample_rate)

    separator = load_speechbrain_model()
    est_sources = separator.separate_file(path=input_wav_path)
    enhanced = est_sources[0].squeeze().cpu().numpy()
    sf.write(enhanced_wav_path, enhanced, sample_rate)

    with open(enhanced_wav_path, "rb") as f:
        enhanced_bytes = f.read()
    return sr.AudioData(enhanced_bytes, sample_rate, 2)

# --- Listen Utility ---
def listen():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=16000) as source:
            st.info("Listening…")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=7)
        if st.session_state.get("use_speechbrain_checkbox", False):
            audio = enhance_audio(audio)
        response = recognizer.recognize_google(audio)
        st.success(f"You said: {response}")
        return response.lower()
    except Exception:
        st.warning("Could not understand audio.")
        return ""

def listen_with_retry(retries=2):
    for _ in range(retries):
        result = listen()
        if result:
            return result
        speak("Please say it again.")
    return ""

# --- PDF & Question Generator ---
@st.cache_resource
def get_question_generator():
    return pipeline("text2text-generation", model="google/flan-t5-large")

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def generate_questions_from_text(text):
    gen = get_question_generator()
    prompt = f"Generate 5 short, direct-answer quiz questions from this content:\n{text}"
    output = gen(prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
    lines = output.strip().split('\n')
    questions = []
    for line in lines:
        line = line.strip()
        for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', 'Question 1:', 'Question 2:', 'Question 3:', 'Question 4:', 'Question 5:']:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if line:
            questions.append(line.rstrip('.'))
    return questions[:5]

# --- PDF Audio Quiz ---
def run_pdf_quiz():
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting content and generating questions..."):
            content = extract_text_from_pdf(uploaded_file)
            questions = generate_questions_from_text(content)
        if not questions:
            st.error("No questions generated. Please try a different PDF or ensure it has extractable text.")
            return

        score = 0
        total_questions = len(questions)

        for i, q in enumerate(questions, start=1):
            speak(f"Question {i}: {q}")
            st.write(f"Q{i}: {q}")
            speak("Your answer?")
            ans = listen_with_retry()
            
            st.info(f"You answered: '{ans}'")
            # This is a dummy check. In a real quiz, you'd compare to a generated answer.
            if ans and "example_correct_keyword" in ans: # Replace with actual answer checking logic
                score += 1
                st.success("Correct!")
            else:
                score -= 0.25
                st.warning("Incorrect.")
        
        speak(f"Your final score is {score:.2f} out of {total_questions}.")
        st.success(f"Quiz completed! Score: {score:.2f}/{total_questions}")

# --- MongoDB Access ---
def load_courses():
    return list(courses_col.find({}, {"_id": 0}))

def load_recommendations():
    return [r["text"] for r in recs_col.find({}, {"_id": 0})]

def insert_course_into_db(course, progress, status):
    courses_col.insert_one({"Course": course, "Progress": progress, "Status": status})

def insert_recommendation(text):
    recs_col.insert_one({"text": text})

# --- User Management ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(hashed_password, user_password):
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password.encode('utf-8'))

def register_user(username, password, role):
    if users_col.find_one({"username": username}):
        return False, "Username already exists."
    
    hashed_password = hash_password(password)
    users_col.insert_one({"username": username, "password": hashed_password, "role": role})
    return True, "Registration successful!"

def authenticate_user(username, password, role):
    user = users_col.find_one({"username": username, "role": role})
    if user and check_password(user["password"], password):
        return True
    return False

# --- Student Pages ---
def student_dashboard():
    st.subheader("📚 Ongoing Courses")
    df = pd.DataFrame(load_courses())
    st.dataframe(df, use_container_width=True)

    st.subheader("🎯 Recommendations")
    for r in load_recommendations():
        st.markdown(f"- {r}")

def student_quiz():
    st.title("🧠 Smart Audio Quiz")
    st.checkbox("Enable SpeechBrain Noise Reduction", key="use_speechbrain_checkbox")
    run_pdf_quiz()

# --- RAG Components for Doubt Clarification ---
@st.cache_resource(show_spinner=False)
def get_rag_models():
    # Model for general text generation (answering questions)
    qa_model_name = "google/flan-t5-large"
    qa_pipeline = pipeline("text2text-generation", model=qa_model_name)

    # Model for generating embeddings (for searching relevant text)
    # Using a smaller, efficient sentence transformer model for embeddings
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name)
    
    return qa_pipeline, embedding_tokenizer, embedding_model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with st.spinner("Generating embedding..."):
        outputs = model(**inputs)
    # Use the mean of the last hidden states as the sentence embedding
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks of given token size with overlap."""
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Simple split by sentences or paragraphs first
    sentences = text.split('.') # Or text.split('\n\n') for paragraphs

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) <= chunk_size:
            current_chunk.extend(words)
            current_length += len(words)
        else:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = words # Start new chunk with current words
            current_length = len(words)
            # Add overlap (simple word-based overlap)
            if overlap > 0 and len(chunks) > 0:
                current_chunk = chunks[-1].split()[-overlap:] + current_chunk
                current_length = len(current_chunk) # Recalculate length for overlap
    
    if current_chunk: # Add the last chunk
        chunks.append(" ".join(current_chunk).strip())
        
    # Filter out very short or empty chunks
    return [chunk for chunk in chunks if len(chunk.split()) > 10]


def student_doubt_clarification():
    st.title("💡 Doubt Clarification")
    st.write("Upload a PDF file and ask questions to clarify your doubts based on its content!")

    qa_pipeline, embedding_tokenizer, embedding_model = get_rag_models()

    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = None

    uploaded_pdf_for_doubt = st.file_uploader("Upload a PDF file for context", type="pdf", key="doubt_pdf_uploader")
    
    if uploaded_pdf_for_doubt and (not st.session_state.document_chunks or st.session_state.uploaded_pdf_name != uploaded_pdf_for_doubt.name):
        st.session_state.uploaded_pdf_name = uploaded_pdf_for_doubt.name
        with st.spinner("Processing PDF: Extracting text, chunking, and creating embeddings. This may take a few moments for large files..."):
            full_text = extract_text_from_pdf(uploaded_pdf_for_doubt)
            
            # Simple chunking for now. For very large docs, consider more sophisticated chunking.
            st.session_state.document_chunks = chunk_text(full_text, chunk_size=300, overlap=50) # Adjust chunk_size/overlap
            
            if st.session_state.document_chunks:
                # Generate embeddings for all chunks
                st.session_state.chunk_embeddings = np.vstack([
                    get_embedding(chunk, embedding_tokenizer, embedding_model)
                    for chunk in st.session_state.document_chunks
                ])
                st.success(f"PDF processed! {len(st.session_state.document_chunks)} chunks created. You can now ask questions.")
            else:
                st.error("Could not extract meaningful content or create chunks from the PDF. Please try a different file.")
                st.session_state.chunk_embeddings = None
                st.session_state.document_chunks = []

    # Initialize chat history
    if "doubt_chat_history" not in st.session_state:
        st.session_state.doubt_chat_history = []
    
    # Display chat history
    for q, a in reversed(st.session_state.doubt_chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

    user_question = st.chat_input("Ask your doubt here (e.g., 'What is recursion?' or 'Explain the concept of neural networks from the document.'):", key="doubt_input")
    
    if user_question:
        st.session_state.doubt_chat_history.append((user_question, "Thinking...")) # Placeholder

        retrieved_context = ""
        if st.session_state.chunk_embeddings is not None and len(st.session_state.chunk_embeddings) > 0:
            with st.spinner("Searching for relevant information in the document..."):
                question_embedding = get_embedding(user_question, embedding_tokenizer, embedding_model)
                similarities = cosine_similarity(question_embedding, st.session_state.chunk_embeddings)[0]
                
                # Get top N most similar chunks
                top_n = 3 # Number of top relevant chunks to retrieve
                top_chunk_indices = similarities.argsort()[-top_n:][::-1]
                
                # Combine relevant chunks, ensuring they fit within the model's context window
                context_chunks = []
                max_qa_tokens = 512 # Max tokens for Flan-T5 input (adjust if model supports more)
                current_context_tokens = 0
                
                for idx in top_chunk_indices:
                    chunk = st.session_state.document_chunks[idx]
                    chunk_tokens = embedding_tokenizer.encode(chunk) # Use embedding tokenizer for simplicity
                    if current_context_tokens + len(chunk_tokens) <= max_qa_tokens - len(embedding_tokenizer.encode(user_question)) - 50: # -50 for prompt overhead
                        context_chunks.append(chunk)
                        current_context_tokens += len(chunk_tokens)
                    else:
                        break # Stop adding chunks if context limit is reached
                
                retrieved_context = "\n\n".join(context_chunks)
                if not retrieved_context:
                    st.warning("No relevant chunks found or could be fit within context window. Answering from general knowledge.")
        
        # Craft the prompt for the QA model
        if retrieved_context:
            prompt = (
                f"You are an intelligent tutor. Based EXCLUSIVELY on the following provided context, "
                f"explain the concept or answer the question. If the information is not present "
                f"in the provided context, state 'I cannot find information about that in the document.'\n\n"
                f"Context: {retrieved_context}\n\n"
                f"Question: {user_question}\n\n"
                f"Explanation:"
            )
        else:
            # Fallback to general knowledge if no context is retrieved or PDF not uploaded
            prompt = (
                f"You are an intelligent tutor. Explain the following concept or answer the question concisely.\n\n"
                f"Question: {user_question}\n\n"
                f"Explanation:"
            )
        
        try:
            with st.spinner("Generating explanation... This might take a moment."):
                bot_reply = ""
                # Call the QA pipeline
                result = qa_pipeline(prompt, max_new_tokens=400, do_sample=True, temperature=0.7, repetition_penalty=1.2)
                bot_reply = result[0]['generated_text'].strip()
                
            if not bot_reply or len(bot_reply.split()) < 5:
                bot_reply = "I'm sorry, I couldn't generate a clear explanation. This might be due to the complexity of the question, lack of specific information in the document, or issues with processing. Please try rephrasing or asking a more specific question."
            
        except Exception as e:
            bot_reply = f"An error occurred during explanation generation: {str(e)}. Please try again or rephrase your question."
        
        # Update the last entry in history
        if st.session_state.doubt_chat_history:
            st.session_state.doubt_chat_history[-1] = (user_question, bot_reply)
        
        st.rerun()

# The original `student_chatbot` function is now a separate "General Chatbot" for non-doubt questions.
def student_general_chatbot():
    st.title("💬 General Chatbot")
    st.write("Ask me anything about general learning tips, career advice, or just chat!")

    if "general_chat_history" not in st.session_state:
        st.session_state.general_chat_history = []
    
    # Use the same QA pipeline for general chat as well
    general_chatbot_pipeline, _, _ = get_rag_models() 
    
    for q, a in reversed(st.session_state.general_chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

    user_input = st.chat_input("Ask me anything!", key="general_input")
    
    if user_input:
        st.session_state.general_chat_history.append((user_input, "Thinking..."))

        context = " ".join([q for q, a in st.session_state.general_chat_history[-4:] if a != "Thinking..."])

        try:
            # Prompt for general conversation
            prompt = f"Context: {context}\nConversation: User: {user_input}\nBot:"
            result = general_chatbot_pipeline(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
            bot_reply = result[0]['generated_text'].strip()
            if not bot_reply:
                bot_reply = "I'm sorry, I couldn't generate a clear answer."
        except Exception as e:
            bot_reply = f"Error processing your request: {str(e)}"
        
        if st.session_state.general_chat_history:
            st.session_state.general_chat_history[-1] = (user_input, bot_reply)
        
        st.rerun()


# --- Faculty Pages ---
def faculty_dashboard():
    st.title("📊 Faculty Dashboard")
    df = pd.DataFrame(load_courses())
    st.dataframe(df, use_container_width=True)

def add_recommendation_page():
    st.subheader("📝 Add Recommendation")
    with st.form("add_rec"):
        text = st.text_area("Recommendation text")
        if st.form_submit_button("Submit") and text:
            insert_recommendation(text)
            st.success("Added!")

def add_course_page():
    st.subheader("➕ Add New Course")
    
    with st.form("add_course_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            course_name = st.text_input("Course Title", placeholder="e.g., Advanced Python")
        with col2:
            progress_val = st.slider("Initial Progress (%)", 0, 100, 0)
        
        course_status = st.selectbox(
            "Course Status",
            ["Not Started", "In Progress", "Completed"],
            index=0 # Default to Not Started
        )

        st.markdown("---") # Visual separator

        submit_button = st.form_submit_button("Add Course to Database")

        if submit_button:
            if course_name:
                insert_course_into_db(course_name, f"{progress_val}%", course_status)
                st.success(f"Course '{course_name}' added successfully with {progress_val}% progress and status '{course_status}'.")
            else:
                st.error("Please provide a Course Title.")

# --- Registration Page ---
def registration_page():
    st.title("Sign Up")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    new_role = st.radio("Register as", ["Student", "Faculty"])

    if st.button("Register"):
        if not new_username or not new_password or not confirm_password:
            st.error("Please fill in all fields.")
        elif new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, message = register_user(new_username, new_password, new_role)
            if success:
                st.success(message + " You can now log in.")
                st.info("Redirecting to login page...")
                time.sleep(2)
                st.session_state.show_login = True
                st.rerun()
            else:
                st.error(message)

# --- Login Page ---
def login_page():
    st.title("🔐 Login")
    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")
    login_role = st.radio("Login as", ["Student", "Faculty"])

    if st.button("Login"):
        if authenticate_user(login_username, login_password, login_role):
            st.session_state.logged_in = True
            st.session_state.username = login_username
            st.session_state.role = login_role
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password for the selected role.")
    
    st.markdown("---")
    st.subheader("New User?")
    if st.button("Register Here"):
        st.session_state.show_login = False
        st.rerun()

# --- Main App ---
def main():
    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "show_login" not in st.session_state:
        st.session_state.show_login = True
    if "uploaded_pdf_name" not in st.session_state: # Track current uploaded PDF name
        st.session_state.uploaded_pdf_name = None

    if not st.session_state.logged_in:
        if st.session_state.show_login:
            login_page()
        else:
            registration_page()
    else:
        # Global Logout Button (Visible for all logged-in users)
        st.sidebar.title("Navigation")
        if st.sidebar.button("Logout", help="Click to log out of the application"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.username = None
            # Clear all relevant chat histories and RAG data on logout
            st.session_state.chat_history = [] 
            st.session_state.doubt_chat_history = []
            st.session_state.general_chat_history = [] 
            st.session_state.document_chunks = [] # Clear document chunks
            st.session_state.chunk_embeddings = None # Clear chunk embeddings
            st.session_state.uploaded_pdf_name = None # Clear uploaded PDF name
            st.session_state.show_login = True
            st.rerun()

        st.sidebar.markdown(f"**Logged in as: {st.session_state.username} ({st.session_state.role})**")

        if st.session_state.role == "Student":
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Dashboard", "🎧 Audio Quiz", "💡 Doubt Clarification", "💬 General Chatbot"])
            with tab1:
                student_dashboard()
            with tab2:
                student_quiz()
            with tab3:
                student_doubt_clarification() # Enhanced doubt clarification chatbot
            with tab4:
                student_general_chatbot() # General chatbot
        elif st.session_state.role == "Faculty":
            tab1, tab2, tab3 = st.tabs(["📋 Dashboard", "➕ Add Recommendation", "📚 Add Course"])
            with tab1:
                faculty_dashboard()
            with tab2:
                add_recommendation_page()
            with tab3:
                add_course_page()

if __name__ == "__main__":
    main()