from flask import Flask, render_template, request, jsonify
from recommender import MovieRecommender
import pandas as pd

app = Flask(__name__)

# Initialize recommender
recommender = MovieRecommender('data/tmdb_5000_movies.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    recommendations = recommender.recommend(movie_title)
    return jsonify(recommendations)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    results = recommender.search_movies(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
