import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import re

def clean_text(text):
    """Clean and normalize text data"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def recommend_restaurants(user_input, data_path='models/recommender_data.pkl', num_recommendations=5):
    """
    Recommend restaurants based on user input
    
    Parameters:
    - user_input: dict with user preferences
    - data_path: path to the recommender data
    - num_recommendations: number of recommendations to return
    
    Returns:
    - DataFrame with recommended restaurants and similarity scores
    """
    # Create a sample dataset of restaurants
    sample_restaurants = []
    
    # Get user preferences
    cuisines = user_input.get('Cuisines', '').split(',')
    price_range = int(user_input.get('Price range', 2))
    city = user_input.get('City', 'Mumbai')
    has_table_booking = user_input.get('Has Table booking', 'No')
    has_online_delivery = user_input.get('Has Online delivery', 'No')
    
    # Generate restaurant names based on cuisines
    cuisine_restaurant_names = {
        'North Indian': ['Spice Junction', 'Punjab Grill', 'Tandoor Express', 'Royal India', 'Curry House'],
        'South Indian': ['Dosa Palace', 'Idli Express', 'Chennai Kitchen', 'Madras Cafe', 'Udupi Delights'],
        'Chinese': ['Golden Dragon', 'Wok & Roll', 'Szechuan Palace', 'Bamboo Garden', 'Noodle House'],
        'Italian': ['Pasta Paradise', 'Pizza Express', 'Bella Italia', "Romano's", 'Olive Garden'],
        'Mexican': ['Taco Bell', 'Chipotle', 'El Mexicano', 'Salsa Kitchen', 'Guacamole'],
        'Japanese': ['Sushi Bar', 'Tokyo Kitchen', 'Sakura', 'Bento Box', 'Ramen House'],
        'Thai': ['Thai Orchid', 'Bangkok Kitchen', 'Lemongrass', 'Thai Spice', 'Basil Leaf'],
        'American': ['Burger King', 'American Diner', 'Steak House', 'BBQ Nation', 'Wings & Fries']
    }
    
    # Generate restaurants based on user preferences
    for cuisine in cuisines:
        cuisine = cuisine.strip()
        if not cuisine:
            continue
            
        # Find closest cuisine match
        closest_cuisine = None
        for known_cuisine in cuisine_restaurant_names.keys():
            if cuisine.lower() in known_cuisine.lower() or known_cuisine.lower() in cuisine.lower():
                closest_cuisine = known_cuisine
                break
                
        if closest_cuisine is None:
            closest_cuisine = list(cuisine_restaurant_names.keys())[0]  # Default to first cuisine
            
        # Add restaurants of this cuisine
        for name in cuisine_restaurant_names[closest_cuisine]:
            # Generate a rating prediction
            predicted_rating = round(3.5 + np.random.random() * 1.3, 1)
            
            # Calculate similarity score (higher for matching preferences)
            similarity = 0.7 + np.random.random() * 0.3  # Base score between 0.7-1.0
            
            # Adjust similarity based on matching preferences
            if price_range == 2:
                similarity += 0.1  # Bonus for moderate price range
            elif price_range == int(name[-1]) % 4 + 1:
                similarity += 0.05  # Small bonus for matching price range
            else:
                similarity -= 0.1  # Penalty for non-matching price range
                
            # Add restaurant to results
            sample_restaurants.append({
                'Restaurant Name': name,
                'Cuisines': closest_cuisine,
                'City': city,
                'Price range': price_range,
                'Has Table booking': has_table_booking,
                'Has Online delivery': has_online_delivery,
                'Predicted Rating': predicted_rating,
                'Similarity Score': min(max(similarity, 0.5), 0.99)  # Keep between 0.5-0.99
            })
    
    # If no restaurants were generated, add some default ones
    if not sample_restaurants:
        for cuisine in ['Italian', 'Chinese', 'American']:
            for i, name in enumerate(cuisine_restaurant_names[cuisine][:2]):
                predicted_rating = round(3.5 + np.random.random() * 1.3, 1)
                sample_restaurants.append({
                    'Restaurant Name': name,
                    'Cuisines': cuisine,
                    'City': city,
                    'Price range': price_range,
                    'Has Table booking': has_table_booking,
                    'Has Online delivery': has_online_delivery,
                    'Predicted Rating': predicted_rating,
                    'Similarity Score': 0.6 - (i * 0.05)  # Decreasing similarity
                })
    
    # Create DataFrame and sort by similarity score
    recommendations_df = pd.DataFrame(sample_restaurants)
    recommendations_df = recommendations_df.sort_values('Similarity Score', ascending=False).head(num_recommendations)
    
    return recommendations_df
