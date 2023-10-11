import streamlit as st
from langchain import Chain, GPTNeoModel
import speech_recognition as sr
import pyttsx3
import langid

# Create a GPT-Neo model
model = GPTNeoModel()

# Create a LangChain chain
chain = Chain(model)

# Create a speech recognition object
r = sr.Recognizer()

# Create a text-to-speech object
engine = pyttsx3.Engine()

# Create a language identification object
language_identifier = langid.LanguageIdentifier.from_modelstring(langid.modelstring_fasttext())

# Define the app's title and description
st.title("Real-Time Speech Translation Tool")
st.write("This is a real-time speech translation tool built with LangChain, GPT-Neo, PyTTX3, and langid.")

# Get the user's input
target_language = st.text_input("Enter the target language:")

# Listen to the user's speech
with sr.Microphone() as source:
    audio = r.listen(source)

# Try to recognize the user's speech
try:
    user_input = r.recognize_google(audio)
    source_language = language_identifier.classify(user_input)[0]
except sr.UnknownValueError:
    st.write("Could not recognize speech")
    exit()

# Translate the user's speech
translation = chain.translate(user_input, source_language, target_language)

# Speak the translation to the user
engine.say(translation)
engine.runAndWait()

# Display the translation to the user
st.write(translation)
