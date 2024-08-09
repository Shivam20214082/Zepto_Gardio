import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator
import gradio as gr
from functools import lru_cache
import json
import requests

# Load the dataset
df = pd.read_csv("cleaned_data.csv")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Define a normalization function for text
def normalize_text_simple(text):
    if pd.isna(text):  # Handle NaNs
        return ''
    text = str(text)  # Convert to string
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Normalize and prepare the separate features
df['normalized_product_name'] = df['product_name'].apply(normalize_text_simple)
df['normalized_product_category'] = df['product_category_tree'].apply(normalize_text_simple)
df['normalized_brand'] = df['brand'].apply(normalize_text_simple)
df['normalized_description'] = df['description'].apply(normalize_text_simple)
df['normalized_extracted_specifications'] = df['extracted_specifications'].apply(normalize_text_simple)

# Initialize the TF-IDF Vectorizers with reduced features
tfidf_name = TfidfVectorizer(max_features=3000)  # Limiting to top 3000 features for better performance
tfidf_category = TfidfVectorizer(max_features=3000)
tfidf_brand = TfidfVectorizer(max_features=3000)
tfidf_description = TfidfVectorizer(max_features=3000)
tfidf_specifications = TfidfVectorizer(max_features=3000)

# Fit and transform the features
tfidf_name_matrix = tfidf_name.fit_transform(df['normalized_product_name'])
tfidf_category_matrix = tfidf_category.fit_transform(df['normalized_product_category'])
tfidf_brand_matrix = tfidf_brand.fit_transform(df['normalized_brand'])
tfidf_description_matrix = tfidf_description.fit_transform(df['normalized_description'])
tfidf_specifications_matrix = tfidf_specifications.fit_transform(df['normalized_extracted_specifications'])

# Define weights for each feature
weights = {
    'name': 0.4,
    'category': 0.2,
    'brand': 0.15,
    'description': 0.15,
    'specifications': 0.1
}

@lru_cache(maxsize=100)
def process_query(query):
    # Detect and translate the query if it's not in English
    detected_lang = detect(query)
    if detected_lang != 'en':
        query = GoogleTranslator(source=detected_lang, target='en').translate(query)

    # Normalize the query
    query = normalize_text_simple(query)
    return query

def get_weighted_tfidf_vector(query):
    query = normalize_text_simple(query)
    
    # Transform the query for each feature
    query_name_vector = tfidf_name.transform([query])
    query_category_vector = tfidf_category.transform([query])
    query_brand_vector = tfidf_brand.transform([query])
    query_description_vector = tfidf_description.transform([query])
    query_specifications_vector = tfidf_specifications.transform([query])
    
    # Compute weighted sum of feature vectors
    weighted_vector = (weights['name'] * query_name_vector +
                       weights['category'] * query_category_vector +
                       weights['brand'] * query_brand_vector +
                       weights['description'] * query_description_vector +
                       weights['specifications'] * query_specifications_vector)
    return weighted_vector

def calculate_discount_percentage(row):
    if row['retail_price'] > 0:
        return 100 * (1 - row['discounted_price'] / row['retail_price'])
    return float('nan')  # Return NaN if the retail_price is 0 or missing

def get_image_urls(image_str):
    try:
        image_urls = json.loads(image_str)  # Safer alternative to eval
    except (json.JSONDecodeError, TypeError):
        image_urls = []  # Handle errors and fallback
    return image_urls

def get_relevant_products(query, max_price, min_discount, min_rating, top_n=12):
    if not query.strip():  # Handle empty queries
        return []  # Return an empty list if query is empty
    query_vector = get_weighted_tfidf_vector(query)
    
    # Compute cosine similarity for each feature matrix
    similarities_name = cosine_similarity(query_vector, tfidf_name_matrix).flatten()
    similarities_category = cosine_similarity(query_vector, tfidf_category_matrix).flatten()
    similarities_brand = cosine_similarity(query_vector, tfidf_brand_matrix).flatten()
    similarities_description = cosine_similarity(query_vector, tfidf_description_matrix).flatten()
    similarities_specifications = cosine_similarity(query_vector, tfidf_specifications_matrix).flatten()
    
    # Combine similarities with weights
    combined_similarities = (weights['name'] * similarities_name +
                             weights['category'] * similarities_category +
                             weights['brand'] * similarities_brand +
                             weights['description'] * similarities_description +
                             weights['specifications'] * similarities_specifications)
    
    top_indices = combined_similarities.argsort()[-top_n:][::-1]  # Get indices of top_n products
    results = df.iloc[top_indices].copy()

    # Convert 'product_rating' to numeric, forcing errors to NaN
    results['product_rating'] = pd.to_numeric(results['product_rating'], errors='coerce')

    # Calculate discount percentages
    results['discount_percentage'] = results.apply(calculate_discount_percentage, axis=1)

    # Apply additional filters considering missing values
    filtered_results = results[
        ((results['discounted_price'].fillna(max_price + 1) <= max_price) | results['discounted_price'].isna()) &
        ((results['discount_percentage'].fillna(-1) >= min_discount) | results['discount_percentage'].isna()) &
        ((results['product_rating'].fillna(-1) >= min_rating) | results['product_rating'].isna())
    ]

    products = []
    for _, row in filtered_results.iterrows():
        # Ensure that 'image' column contains publicly accessible URLs
        image_urls = get_image_urls(row['image'])
        if image_urls:  # If the list is not empty
            products.append({
                "image": image_urls[0],  # Use the first image URL
                "name": row['product_name'],
                "price": row['discounted_price'],
                "url": row['product_url']
            })
    return products

def search_products(query, max_price, min_discount, min_rating):
    results = get_relevant_products(query, max_price, min_discount, min_rating)
    if not results:
        return "No results found for the given query."
    
    # Create the output HTML string with vertical layout (one product per row)
    output_html = """
    <div style="display: flex; flex-direction: column; align-items: center;">
    """
    for product in results:
        output_html += f"""
        <div style="width: 80%; margin-top: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 10px; background-color: #f9f9f9; text-align: center;">
            <img src="{product['image']}" width="150" style="object-fit: cover; border-radius: 10px;" onerror="this.src='https://via.placeholder.com/150';">
            <div style="font-size: 18px; font-weight: bold; color: #333; margin-top: 10px;">{product['name']}</div>
            <div style="font-size: 16px; font-weight: bold; color: #e74c3c; margin-top: 5px;">Price: {product['price']}</div>
            <div style="margin-top: 10px;">
                <a href="{product['url']}" style="display: inline-block; padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; border-radius: 5px;">View Product</a>
            </div>
        </div>
        """
    output_html += "</div>"
    return output_html

# Gradio interface setup
interface = gr.Interface(
    fn=search_products,
    inputs=[
        gr.Textbox(placeholder="Type a product name...", lines=1, type="text", label="Search"),
        gr.Slider(minimum=0, maximum=10000, value=1000, label="Max Price"),
        gr.Slider(minimum=0, maximum=100, value=10, label="Min Discount (%)"),
        gr.Slider(minimum=0, maximum=5, value=1, label="Min Rating")
    ],
    outputs="html",
    title="Welcome to Zepto Product Search System",
    description="Enter a search term to find relevant products and filter by price, discount, and rating.",
    examples=[["Shirt", 1000, 10, 1], ["Smartphone", 1500, 5, 4], ["Shoes", 500, 20, 3]],  # Example input lists
    allow_flagging="never"  # Disable flagging
)

# Launch the Gradio interface
interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
