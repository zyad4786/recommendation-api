from flask import Flask, request, jsonify
import pandas as pd
import requests
import json
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)
CORS(app)

# -------------------- Product Recommendation --------------------

# Load products from the local JSON file once at startup
with open('final_clean_productsV2.json', 'r', encoding='utf-8') as file:
    product_data = json.load(file)

# Convert to DataFrame
def preprocess_products(product_data):
    df = pd.DataFrame(product_data)
    
    # Create tags using only ingredients + labels (no categories!)
    df['tags'] = (df['ingredients'].fillna('') + ' ' + df['labels'].fillna('')).str.lower()
    
    return df

products_df = preprocess_products(product_data)

@app.route("/recommend_products", methods=["POST"])
def recommend_products():
    data = request.get_json()
    user_positive_preferences = " ".join(data.get("positive_preferences", []))
    user_negative_preferences = [x.lower() for x in data.get("negative_preferences", [])]
    category_filter = data.get("category_filter", "").lower()

    page = int(data.get("page", 1))
    page_size = int(data.get("page_size", 10))
    start = (page - 1) * page_size
    end = start + page_size

    df = products_df.copy()

    vectorizer = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(df["tags"])
    user_vector = vectorizer.transform([user_positive_preferences.lower()])
    df["similarity_score"] = cosine_similarity(user_vector, item_vectors)[0]

    def contains_negative(tags, negative_list):
        for negative in negative_list:
            if negative in tags and f"{negative}-free" not in tags:
                return True
        return False

    df = df[~df["tags"].apply(lambda x: contains_negative(x, user_negative_preferences))]

    if category_filter:
        df = df[df["category"].str.lower() == category_filter]

    recommended = df.sort_values(by="similarity_score", ascending=False).iloc[start:end]

    return jsonify(recommended.drop(columns=["tags"]).to_dict(orient="records"))

@app.route("/available_categories", methods=["GET"])
def get_available_categories():
    # Just return all categories in the dataset (you said you'll hardcode later but this keeps it for now)
    categories = list(set([
        p.get("category", "unknown").lower()
        for p in product_data if p.get("category")
    ]))
    return jsonify(sorted(categories))

# -------------------- Meal Plan Recommendation --------------------

with open("meal_dataset.json", "r") as file:
    meals = json.load(file)

def contains_meal_negative(tags, negative_list):
    for negative in negative_list:
        if negative in tags and f"{negative}-free" not in tags:
            return True
    return False

@app.route("/mealplan", methods=["POST"])
def generate_meal_plan():
    data = request.get_json()
    user_positive_preferences = " ".join(data.get("positive_preferences", []))
    user_negative_preferences = [x.lower() for x in data.get("negative_preferences", [])]
    calorie_limit = data.get("calorie_limit", 2000)

    df = pd.DataFrame(meals)
    df["tags"] = df["tags"].str.lower()

    vectorizer = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(df["tags"])
    user_vector = vectorizer.transform([user_positive_preferences.lower()])
    df["similarity_score"] = cosine_similarity(user_vector, item_vectors)[0]

    df = df[~df["tags"].apply(lambda x: contains_meal_negative(x, user_negative_preferences))]

    def get_meal_plan(df, calorie_limit):
        selected_meals = {}
        total_calories = 0

        meal_types = ["breakfast", "lunch", "dinner"]
        for meal_type in meal_types:
            top_meals = df[df["category"] == meal_type].sort_values(by="similarity_score", ascending=False).head(10)
            meals_of_type = top_meals.sample(frac=1).reset_index(drop=True)

            for _, meal in meals_of_type.iterrows():
                if total_calories + meal["calories"] <= calorie_limit:
                    selected_meals[meal_type] = {
                        "name": meal.get("name", "Unknown"),
                        "ingredients": meal.get("ingredients", [])
                    }
                    total_calories += meal["calories"]
                    break

        return selected_meals, total_calories

    recommended_meals, total_calories = get_meal_plan(df, calorie_limit)

    return jsonify({
        "recommended_meals": recommended_meals,
        "total_calories": total_calories
    })

if __name__ == "__main__":
    app.run(debug=True)