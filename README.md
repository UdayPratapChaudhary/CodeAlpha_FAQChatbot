# CodeAlpha FAQ Chatbot

A simple FAQ chatbot built with Python, Streamlit, and scikit-learn. It cleans the user's question, converts questions into TF-IDF vectors, and uses cosine similarity to select the closest FAQ answer.

## Features

- Interactive chat interface
- Example FAQs about internships and project submission
- Text preprocessing with regular expressions
- TF-IDF vectorization with unigrams and bigrams
- Cosine similarity matching
- Fallback response for questions that do not match well

## Requirements

- Python 3.9 or newer

## Run locally

1. Open a terminal in this folder.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## How it works

1. Clean each FAQ and the user's query (lowercase, remove punctuation, normalize spaces).
2. Convert FAQ questions into TF-IDF vectors.
3. Convert the user's question into a vector using the same vectorizer.
4. Calculate cosine similarity between the query and each FAQ question.
5. Return the best answer if the similarity passes a small threshold; otherwise show a fallback message.

## Customize the FAQ data

Edit the `FAQS` list in `app.py`. Each item should have a `question` and an `answer`.

## Suggested GitHub repository name

`CodeAlpha_FAQChatbot`

## Submission checklist

- [ ] Install dependencies and run the app.
- [ ] Test several FAQ questions and unrelated questions.
- [ ] Upload the complete source code to GitHub.
- [ ] Post a short video demonstration on LinkedIn, tagging CodeAlpha and adding the repository link.
- [ ] Submit the project using the internship submission form.
