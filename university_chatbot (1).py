import streamlit as st
import json
import logging
import random
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Set up logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Greetings, fallbacks, follow-ups
greetings = [
    "Hi there! I'm your friendly university assistant, ready to help! What's on your mind?",
    "Hello! Excited to assist you with university questions. What's up?",
    "Hey! I'm here to answer your questions about our university. What's first?"
]

fallback_responses = [
    "I'm not sure I caught that! Could you rephrase or ask something else?",
    "Oops, I didn't get that one. Try asking in a different way or check out support@university.edu!",
    "Hmm, that's a new one for me! Can you clarify or ask about something else, like admissions or campus tours?"
]

follow_ups = [
    "Anything else I can help with?",
    "Got more questions? I'm all ears!",
    "What's next on your list?"
]

# Load knowledge base and initialize model
knowledge_base = []
question_embeddings = None
model = None

try:
    with open('qa_dataset.json', 'r') as f:
        knowledge_base = json.load(f)['questions']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('distilbert-base-uncased', device=device)

    question_texts = [item['question'].lower() for item in knowledge_base]
    question_embeddings = model.encode(question_texts, convert_to_tensor=True)

except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
    st.error("Knowledge base file is missing or malformed.")
    logging.error(f"Failed to load knowledge base: {e}")

# Function to get response
def get_response(user_input):
    if not knowledge_base or question_embeddings is None:
        return random.choice(fallback_responses), random.choice(follow_ups)

    user_input = user_input.lower().strip()
    logging.info(f"User query: {user_input}")
    
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    best_score = cos_scores.max().item()
    best_idx = cos_scores.argmax().item()

    if best_score > 0.7:
        response = knowledge_base[best_idx]['answer']
        logging.info(f"Matched: {knowledge_base[best_idx]['question']} | Score: {best_score}")
        return response, random.choice(follow_ups)
    else:
        logging.info(f"No good match found (score={best_score:.2f}), using fallback.")
        return random.choice(fallback_responses), random.choice(follow_ups)

# Streamlit app
def main():
    st.set_page_config(page_title="University Chatbot", layout="centered")
    st.title("🎓 University Chatbot")
    st.write("I'm here to help with questions about admissions, registration, and more. Ask away!")

    st.markdown("👋 Example questions: *What programs do you offer?*, *How do I apply?*, *Where is the campus located?*")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [(None, random.choice(greetings))]
    
    # Reset button
    if st.button("🔄 Reset Chat"):
        st.session_state.chat_history = [(None, random.choice(greetings))]

    # User input
    user_input = st.text_input("Your question:", key="user_input")

    if user_input:
        response, follow_up = get_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.session_state.chat_history.append(("Bot", follow_up))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if hasattr(st, "chat_message"):
            # Streamlit v1.26+ (nicer chat display)
            if speaker == "You":
                with st.chat_message("user"):
                    st.markdown(message)
            elif speaker == "Bot":
                with st.chat_message("assistant"):
                    st.markdown(message)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message)
        else:
            # Fallback for older Streamlit
            if speaker == "You":
                st.markdown(f"**You**: {message}")
            elif speaker == "Bot":
                st.markdown(f"**Bot**: {message}")
            else:
                st.markdown(f"{message}")

if __name__ == "__main__":
    main()
