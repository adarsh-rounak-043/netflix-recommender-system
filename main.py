import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import pickle
import requests
import os
import sys

# Optional debug
print("Running on Python:", sys.version)

# Setup base directory for consistent path usage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load NLP model and vectorizer
filename = os.path.join(BASE_DIR, 'nlp_model.pkl')
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'tranform.pkl'), 'rb'))

# Declare globals
data = None
similarity = None

def create_similarity():
    data = pd.read_csv(os.path.join(BASE_DIR, 'main_data.csv'))
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    global data, similarity
    m = m.lower()
    if data is None or similarity is None:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return 'Sorry! try another movie name'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)[1:11]
        return [data['movie_title'][x[0]] for x in lst]

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    data = pd.read_csv(os.path.join(BASE_DIR, 'main_data.csv'))
    return list(data['movie_title'].str.capitalize())

# Flask App
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions, api_key=os.getenv("API_KEY"))

@app.route("/similarity", methods=["POST"])
def similarity_route():
    movie = request.form['name']
    rc = rcmd(movie)
    if isinstance(rc, str):
        return rc
    else:
        return "---".join(rc)

@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    suggestions = get_suggestions()

    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_ql_2'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = bs.BeautifulSoup(response.text, 'lxml')
    soup_result = soup.find_all('div', class_="ipc-html-content-inner-div")

    reviews_list = []
    reviews_status = []

    for reviews in soup_result:
        if reviews.text:
            reviews_list.append(reviews.text)
            movie_review_list = np.array([reviews.text])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    return render_template('recommend.html', title=title, poster=poster, overview=overview,
                           vote_average=vote_average, vote_count=vote_count,
                           release_date=release_date, runtime=runtime, status=status,
                           genres=genres, movie_cards=movie_cards, reviews=movie_reviews,
                           casts=casts, cast_details=cast_details)

if __name__ == '__main__':
    app.run()
