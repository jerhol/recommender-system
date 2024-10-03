import os
from flask import Flask, render_template, request, jsonify
from RecommenderSystem import RecommenderSystem

app = Flask(__name__)
recommender = RecommenderSystem()
user_rated_books = []

@app.route('/')
def index():
    book_titles = recommender.get_books()
    return render_template('index.html', book_titles=book_titles, user_rated_books=user_rated_books)

@app.route('/rate', methods=['POST'])
def rate_book():
    book_title = request.form['book_title']
    book_rating = request.form['book_rating']
    user_rated_books.append((book_title, int(book_rating)))
    return render_template('_rated_books.html', user_rated_books=user_rated_books)

@app.route('/clear', methods=['POST'])
def clear_ratings():
    user_rated_books.clear()
    return render_template('_rated_books.html', user_rated_books=user_rated_books)

@app.route('/recommend', methods=['GET'])
def recommend():
    if not user_rated_books:
        return jsonify([])

    recommendations = recommender.recommend_books(rated_books=user_rated_books, user_id="999999", n=20)
    return render_template('_recommendations.html', recommendations=recommendations)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
