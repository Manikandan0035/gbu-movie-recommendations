import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data (replace with your actual file paths)
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocess data
def prepare_data():
    # Calculate average rating for each movie
    movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    movie_ratings.columns = ['movieId', 'avg_rating']
    
    # Merge with movie data
    movies_with_ratings = movies.merge(movie_ratings, on='movieId')
    
    # Create a TF-IDF matrix based on genres
    tfidf = TfidfVectorizer(stop_words='english')
    movies_with_ratings['genres'] = movies_with_ratings['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies_with_ratings['genres'])
    
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return movies_with_ratings, cosine_sim

movies_df, cosine_sim = prepare_data()

# Recommendation function
def get_recommendations(title, num_recommendations=5):
    # Get the index of the movie that matches the title
    idx = movies_df[movies_df['title'] == title].index[0]
    
    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top similar movies
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top similar movies
    return movies_df.iloc[movie_indices][['title', 'genres', 'avg_rating']]

@app.route('/')
def home():
    # Get list of all movie titles for dropdown
    movie_titles = movies_df['title'].tolist()
    return render_template('index.html', movie_titles=movie_titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    recommendations = get_recommendations(movie_title)
    return render_template('recommendations.html', 
                          movie_title=movie_title,
                          recommendations=recommendations.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)
