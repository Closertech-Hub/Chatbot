import streamlit as st
import json
import logging
import random
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Set up logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load knowledge base (replace with your JSON file path)
try:
    with open('qa_dataset.json', 'r') as f:
        qa_dataset = json.load(f)['questions']
except FileNotFoundError:
    st.error("Knowledge base file not found. Please ensure 'qa_dataset.json' is in the same directory.")
    qa_dataset = []

# Initialize sentence transformer for semantic similarity
model = SentenceTransformer('distilbert-base-uncased')

# Precompute embeddings for knowledge base questions
question_texts = [item['question'].lower() for item in knowledge_base]
question_embeddings = model.encode(question_texts, convert_to_tensor=True)

# Fallback responses for unrecognized queries
fallback_responses = [
    "I'm not sure I caught that! Could you rephrase or ask something else?",
    "Oops, I didn't get that one. Try asking in a different way or check out support@university.edu!",
    "Hmm, that's a new one for me! Can you clarify or ask about something else, like admissions or campus tours?"
]

# Greetings for conversation start
greetings = ["Hi there! I'm your friendly university assistant, ready to help! What's on your mind?",
             "Hello! Excited to assist you with university questions. What's up?",
             "Hey! I'm here to answer your questions about our university. What's first?"]

# Follow-up prompts
follow_ups = ["Anything else I can help with?", "Got more questions? I'm all ears!", "What's next on your list?"]

# Process user input and find best match
def get_response(user_input):
    user_input = user_input.lower().strip()
    logging.info(f"User query: {user_input}")

    # Encode user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    best_score = cos_scores.max().item()
    best_idx = cos_scores.argmax().item()

    # Threshold for matching (adjust as needed)
    if best_score > 0.7:  # Higher score means better match
        response = knowledge_base[best_idx]['answer']
        logging.info(f"Matched query: {knowledge_base[best_idx]['question']}, Response: {response}")
        return response, random.choice(follow_ups)
    else:
        logging.info("No match found, using fallback")
        return random.choice(fallback_responses), random.choice(follow_ups)

# Streamlit app
def main():
    st.title("University Chatbot")
    st.write("I'm here to help with questions about admissions, registration, and more! Ask away!")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [(None, random.choice(greetings))]
    if 'first_interaction' not in st.session_state:
        st.session_state.first_interaction = True

    # User input
    user_input = st.text_input("Your question:", key="user_input")

    if user_input:
        # Get response and follow-up
        response, follow_up = get_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.session_state.chat_history.append(("Bot", follow_up))
        st.session_state.first_interaction = False

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You**: {message}")
        elif speaker == "Bot":
            st.markdown(f"**Bot**: {message}")
        else:
            st.markdown(f"{message}")

if __name__ == "__main__":
    main()
