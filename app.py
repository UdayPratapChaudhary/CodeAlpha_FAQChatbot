import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="centered")

FAQS = [
    {
        "question": "What is CodeAlpha?",
        "answer": "CodeAlpha is an organization that provides internship and learning opportunities for students and aspiring developers."
    },
    {
        "question": "How do I apply for an internship?",
        "answer": "Check the official CodeAlpha internship instructions and complete the application process described there."
    },
    {
        "question": "How do I submit my project?",
        "answer": "Upload your complete source code to GitHub using the repository naming format requested in your internship instructions, then submit the project using the provided submission form."
    },
    {
        "question": "What should my GitHub repository be named?",
        "answer": "Use the required format CodeAlpha_ProjectName. For example, CodeAlpha_FAQChatbot."
    },
    {
        "question": "Do I need to post my project on LinkedIn?",
        "answer": "Follow your internship instructions. They may ask you to share your internship status and publish a video explanation of your project on LinkedIn, tagging CodeAlpha and including your GitHub repository link."
    },
    {
        "question": "How many tasks do I need to complete?",
        "answer": "Follow the instructions provided for your internship batch. Your task sheet may require completing two or three projects from the available tasks."
    },
    {
        "question": "Which programming language is used in this project?",
        "answer": "This FAQ chatbot is built with Python, Streamlit, and scikit-learn."
    },
    {
        "question": "What is NLP?",
        "answer": "Natural Language Processing (NLP) is a field of AI that helps computers process, analyze, and work with human language."
    },
    {
        "question": "How does this chatbot find an answer?",
        "answer": "It cleans the text, converts FAQ questions and the user's query into TF-IDF vectors, and uses cosine similarity to find the closest question. If the similarity is too low, it asks the user to rephrase."
    },
    {
        "question": "Is this chatbot always correct?",
        "answer": "No. It uses similarity matching, so it can misunderstand unusual wording. Try rephrasing your question if the answer is not relevant."
    },
]

def clean_text(text):
    """Lowercase text and remove punctuation while keeping letters and digits."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

faq_questions = [item["question"] for item in FAQS]
clean_questions = [clean_text(q) for q in faq_questions]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
faq_vectors = vectorizer.fit_transform(clean_questions)

def get_answer(user_question):
    cleaned = clean_text(user_question)
    if not cleaned:
        return "Please type a question so I can help.", 0.0, None

    user_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(user_vector, faq_vectors)[0]
    best_index = scores.argmax()
    best_score = float(scores[best_index])

    # Threshold prevents the bot from confidently answering unrelated questions.
    if best_score < 0.12:
        return (
            "Sorry, I couldn't find a matching FAQ. Try asking about CodeAlpha, "
            "internship submissions, GitHub, LinkedIn, Python, or NLP.",
            best_score,
            None,
        )
    return FAQS[best_index]["answer"], best_score, FAQS[best_index]["question"]

st.title("🤖 CodeAlpha FAQ Chatbot")
st.write("Ask a question about internships, project submissions, GitHub, or this chatbot.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! 👋 Ask me a question about the internship or this project."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer, score, matched_question = get_answer(user_input)

    if matched_question:
        answer += f"\n\n*Matched FAQ: “{matched_question}”*"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

with st.sidebar:
    st.header("Example questions")
    for item in FAQS:
        st.markdown(f"- {item['question']}")
    st.divider()
    st.caption("Matching method: TF-IDF + cosine similarity")
    st.caption("FAQ answers are examples. Check official internship instructions for current requirements.")
