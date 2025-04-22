from flask import Flask, request, jsonify
import pandas as pd
import requests
import json
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# -------------------- Product Recommendation --------------------
CACHED_CATEGORIES = []

def fetch_product_data():
    url = "https://world.openfoodfacts.org/api/v2/search?country=saudi-arabia&page_size=100&fields=code,product_name,image_front_url,labels,categories,categories_tags,allergens,ingredients_text,nutriments"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        data = response.json()
        products = data["products"]
        extracted_data = []
        for product in products:
            extracted_data.append({
                "product_id": product.get("code", "Unknown"),
                "name": product.get("product_name", "Unknown"),
                "image": product.get("image_front_url", ""),
                "category": product.get("categories_tags", ["Unknown"])[0] if product.get("categories_tags") else "Unknown",
                "ingredients": product.get("ingredients_text", "Not available"),
                "nutritional_info": {
                    "fat": product.get("nutriments", {}).get("fat", "N/A"),
                    "protein": product.get("nutriments", {}).get("proteins", "N/A"),
                    "calories": product.get("nutriments", {}).get("energy-kcal", "N/A"),
                },
                "tags": product.get("labels", "") + " " + product.get("categories", "") + " " + product.get("allergens", "")
            })
        return extracted_data
    else:
        return []

def preprocess_products(product_data):
    df = pd.DataFrame(product_data)
    df["tags"] = df["tags"].str.lower()
    return df

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

    products = fetch_product_data()
    if not products:
        return jsonify({"error": "Failed to retrieve products"}), 500

    df = preprocess_products(products)

    vectorizer = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(df["tags"])
    user_vector = vectorizer.transform([user_positive_preferences.lower()])
    df["similarity_score"] = cosine_similarity(user_vector, item_vectors)[0]

    def contains_negative(tags, negative_list):
        for negative in negative_list:
            if negative in tags and f"{negative}-free" not in tags:
                return True
        return False

    filtered_df = df[~df["tags"].apply(lambda x: contains_negative(x, user_negative_preferences))]

    if category_filter:
        filtered_df = filtered_df[filtered_df["category"].str.lower() == category_filter]

    recommended = filtered_df.sort_values(by="similarity_score", ascending=False).iloc[start:end]

    return jsonify(recommended.to_dict(orient="records"))

@app.route("/available_categories", methods=["GET"])
def get_available_categories():
    global CACHED_CATEGORIES
    if not CACHED_CATEGORIES:
        try:
            products = fetch_product_data()
            categories = list(set([
                p.get("category", "unknown").lower()
                for p in products if p.get("category")
            ]))
            CACHED_CATEGORIES = sorted(categories)
        except Exception as e:
            return jsonify({"error": f"Failed to load categories: {str(e)}"}), 500

    return jsonify(CACHED_CATEGORIES)

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
            meals_of_type = df[df["category"] == meal_type].sort_values(by="similarity_score", ascending=False)
            for _, meal in meals_of_type.iterrows():
                if total_calories + meal["calories"] <= calorie_limit:
                    selected_meals[meal_type] = {
                        "name": meal["name"],
                        "ingredients": meal["ingredients"]
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
