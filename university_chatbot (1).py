import streamlit as st
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging

# Set up logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Sample knowledge base (replace with university-specific FAQs)
knowledge_base = {
    "admission requirements": "To apply, you need a high school diploma, SAT/ACT scores, and a completed application form. Check the university website for specific GPA requirements.",
    "registration deadline": "The registration deadline for the upcoming semester is August 15, 2025. Late registration may incur a fee.",
    "financial aid": "Financial aid applications are due by June 30, 2025. Visit the financial aid office or website for forms and eligibility details.",
    "campus tour": "Campus tours are available weekly. Register online at the university's admissions page.",
    "contact support": "Contact the student support office at support@university.edu or call (123) 456-7890."
}

# Preprocess user input
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Match user query to knowledge base
def get_response(user_input):
    processed_input = preprocess_text(user_input)
    best_match = None
    max_score = 0

    for question, answer in knowledge_base.items():
        processed_question = preprocess_text(question)
        score = len(set(processed_input) & set(processed_question))
        if score > max_score:
            max_score = score
            best_match = answer

    if max_score > 0:
        logging.info(f"Query: {user_input}, Response: {best_match}")
        return best_match
    else:
        logging.info(f"Query: {user_input}, Response: Fallback")
        return "Sorry, I don't understand your question. Please try rephrasing or contact support@university.edu."

# Streamlit app
def main():
    st.title("University Chatbot")
    st.write("Ask about admissions, registration, financial aid, or campus tours!")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("Your question:", key="user_input")

    if user_input:
        response = get_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.write(f"**You**: {message}")
        else:
            st.write(f"**Bot**: {message}")

if __name__ == "__main__":
    main()