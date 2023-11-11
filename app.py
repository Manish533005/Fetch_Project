from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
brand_category_df = pd.read_csv('brand_category.csv')
categories_df = pd.read_csv('categories.csv')
offer_retailer_df = pd.read_csv('offer_retailer.csv')

# Merge datasets
merged_df = pd.merge(offer_retailer_df, brand_category_df, on='BRAND', how='left')
merged_df = pd.merge(merged_df, categories_df, on='PRODUCT_CATEGORY', how='left')

# Text preprocessing and feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df['OFFER'])

# Function to get relevant offers based on user input
def get_relevant_offers(user_input):
    user_tfidf = tfidf_vectorizer.transform([user_input])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Combine scores with offer details
    merged_df['similarity_score'] = similarity_scores

    # Sort by similarity score
    sorted_offers = merged_df.sort_values(by='similarity_score', ascending=False)

    return sorted_offers[['OFFER', 'RETAILER', 'BRAND', 'PRODUCT_CATEGORY', 'similarity_score']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    user_query = request.form['user_input']
    relevant_offers = get_relevant_offers(user_query)
    return render_template('result.html', offers=relevant_offers)

if __name__ == '__main__':
    app.run(debug=True)
