import streamlit as st
import torch
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import re
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS
import text2emotion as te
import nltk
import os
import subprocess
import numpy as np
import base64
import streamlit.components.v1 as components
import threading
import schedule
import time
from datetime import datetime

nltk.download('punkt', quiet=True)

# Ensure ffmpeg is available
FFMPEG_PATH = r"C:\\ffmpeg\\bin"
if FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH
try:
    subprocess.check_output(["ffmpeg", "-version"])
except FileNotFoundError:
    st.error("❌ ffmpeg not found! Please check your installation or path.")
    st.stop()

# Streamlit setup
st.set_page_config(page_title="Companion Robot", page_icon="🤖", layout="centered")
st.title("🎧 Companion Robot")

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_listening" not in st.session_state:
    st.session_state.active_listening = False
if "stop_listening" not in st.session_state:
    st.session_state.stop_listening = False
if "schedules" not in st.session_state:
    st.session_state.schedules = []
if "reminder_triggered" not in st.session_state:
    st.session_state.reminder_triggered = False
if "latest_audio" not in st.session_state:
    st.session_state.latest_audio = None
if "last_reminder_message" not in st.session_state:
    st.session_state.last_reminder_message = ""

# Load Models
@st.cache_resource
def load_models():
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    whisper_model = whisper.load_model("base")
    return tokenizer, model, whisper_model

tokenizer, bot_model, whisper_model = load_models()

# Helpers
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def detect_emotion(text):
    emotions = te.get_emotion(text)
    return max(emotions, key=emotions.get) if emotions else "neutral"

def speak_response(text, lang='en', emotion='neutral'):
    if lang == 'ur':
        text += "!" if emotion in ["happy", "surprise"] else "."
    elif emotion == "angry":
        text = "😠 " + text.upper()
    elif emotion == "sad":
        text = "😢 " + text.lower()
    elif emotion == "happy":
        text = "😊 " + text.capitalize()

    tts = gTTS(text=text, lang='ur' if lang == 'ur' else 'en')
    tts.save("response.mp3")

    with open("response.mp3", "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        st.session_state.latest_audio = b64
        st.session_state.last_reminder_message = text
        st.session_state.reminder_triggered = True

def recognize_speech_whisper(audio_file):
    if not audio_file or not os.path.exists(audio_file):
        return ""
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except:
        return ""

def record_until_silence(threshold=0.003, silence_duration=1.5, fs=16000):
    buffer_duration = 0.2
    buffer_size = int(fs * buffer_duration)
    silence_chunks_required = int(silence_duration / buffer_duration)

    audio_data = []
    silence_counter = 0
    st.info("🎤 Speak now...")

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
            while True:
                buffer, _ = stream.read(buffer_size)
                buffer = np.squeeze(buffer)
                audio_data.append(buffer)

                rms = np.sqrt(np.mean(buffer**2))
                if rms < threshold:
                    silence_counter += 1
                    if silence_counter >= silence_chunks_required:
                        break
                else:
                    silence_counter = 0

        audio_np = np.concatenate(audio_data)
        audio_np = (audio_np * 32767).astype(np.int16)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(temp_wav.name, fs, audio_np)
        return temp_wav.name
    except:
        return None

def generate_response(user_input, emotion):
    st.session_state.chat_history.append(f"🧑 You: {user_input}")
    context = "\n".join(st.session_state.chat_history[-5:])
    prompt = f"User is feeling [{emotion}]. {context}"

    inputs = tokenizer([prompt], return_tensors="pt")
    reply_ids = bot_model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    st.session_state.chat_history.append(f"🤖 Bot: {response}")
    return response

def handle_conversation():
    audio_path = record_until_silence()
    user_text = recognize_speech_whisper(audio_path)

    if not user_text.strip():
        st.warning("Could not understand anything.")
        return

    user_text_clean = remove_emojis(user_text)
    try:
        lang = detect(user_text_clean)
    except:
        lang = "en"

    try:
        translated_input = GoogleTranslator(source='auto', target='en').translate(user_text_clean) if lang == 'ur' else user_text_clean
    except:
        translated_input = user_text_clean

    emotion = detect_emotion(translated_input)
    response = generate_response(translated_input, emotion)

    try:
        final_response = GoogleTranslator(source='en', target='ur').translate(response) if lang == 'ur' else response
    except:
        final_response = response

    st.subheader("🗣️ You said:")
    st.write(user_text_clean)
    st.subheader("😊 Detected Emotion:")
    st.write(emotion.capitalize())
    st.subheader("🤖 Bot Response:")
    st.write(final_response)
    speak_response(final_response, lang, emotion)

def start_listening_loop():
    while st.session_state.active_listening and not st.session_state.stop_listening:
        handle_conversation()

def start_schedule_loop():
    while True:
        schedule.run_pending()
        time.sleep(1)

def schedule_reminder(text, time_str):
    try:
        datetime.strptime(time_str, "%H:%M")
        schedule.every().day.at(time_str).do(speak_response, text=text)
        st.session_state.schedules.append(f"⏰ Reminder at {time_str}: {text}")
    except ValueError:
        st.warning("Invalid time format. Please use HH:MM (24h).")

with st.sidebar:
    st.markdown("### 🎛️ Control Panel")

    if st.button("▶️ Start Listening"):
        st.session_state.active_listening = True
        st.session_state.stop_listening = False
        threading.Thread(target=start_listening_loop, daemon=True).start()

    if st.button("⏹️ Stop Listening"):
        st.session_state.stop_listening = True
        st.session_state.active_listening = False

    st.markdown("---")
    st.markdown("### ⏰ Set Schedule Reminder")

    reminder_text = st.text_input("Reminder text:")
    reminder_time = st.text_input("Reminder time (HH:MM, 24h)", "10:00")

    if st.button("🔔 Set Reminder"):
        if reminder_text and reminder_time:
            schedule_reminder(reminder_text, reminder_time)
            st.success(f"Reminder set for {reminder_time}")
        else:
            st.warning("Please enter both reminder text and valid time.")

    if st.session_state.schedules:
        st.markdown("### 📋 Active Schedules")
        for s in st.session_state.schedules:
            st.write(s)

threading.Thread(target=start_schedule_loop, daemon=True).start()

if st.session_state.reminder_triggered and st.session_state.latest_audio:
    st.markdown("## 🔔 Reminder Alert!")
    st.markdown(f"**Message:** {st.session_state.last_reminder_message}")

    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{st.session_state.latest_audio}" type="audio/mp3">
        </audio>
    """
    components.html(audio_html, height=0)
    st.session_state.reminder_triggered = False

if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for msg in st.session_state.chat_history:
        st.write(msg)
