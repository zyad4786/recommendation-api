
from flask import Flask, request, jsonify
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------- Product Recommendation --------------------

def fetch_product_data():
    url = "https://world.openfoodfacts.org/api/v2/search?country=saudi-arabia&page_size=100"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        products = data["products"]
        extracted_data = []
        for product in products:
            extracted_data.append({
                "product_id": product.get("code", "Unknown"),
                "name": product.get("product_name", "Unknown"),
                "image": product.get("image_front_url", ""),  # Image support
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
    user_positive_preferences = data.get("positive_preferences", "")
    user_negative_preferences = [x.lower() for x in data.get("negative_preferences", [])]

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
    recommended = filtered_df.sort_values(by="similarity_score", ascending=False).head(10)
    return jsonify(recommended[["product_id", "name", "image", "similarity_score"]].to_dict(orient="records"))

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
    user_positive_preferences = data.get("positive_preferences", "")
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
        selected_meals = {"breakfast": None, "lunch": None, "dinner": None}
        total_calories = 0
        grocery_list = set()

        meal_options = {
            "breakfast": df[df["category"] == "breakfast"].sort_values(by="similarity_score", ascending=False).to_dict(orient="records"),
            "lunch": df[df["category"] == "lunch"].sort_values(by="similarity_score", ascending=False).to_dict(orient="records"),
            "dinner": df[df["category"] == "dinner"].sort_values(by="similarity_score", ascending=False).to_dict(orient="records"),
        }

        for meal_type in ["breakfast", "lunch", "dinner"]:
            for meal in meal_options[meal_type]:
                if total_calories + meal["calories"] <= calorie_limit:
                    selected_meals[meal_type] = meal["name"]
                    total_calories += meal["calories"]
                    grocery_list.update(meal["ingredients"])
                    break

        return selected_meals, total_calories, list(grocery_list)

    recommended_meals, total_calories, grocery_list = get_meal_plan(df, calorie_limit)

    return jsonify({
        "recommended_meals": recommended_meals,
        "total_calories": total_calories,
        "grocery_list": grocery_list
    })

if __name__ == "__main__":
    app.run(debug=True)