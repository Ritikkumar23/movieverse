import streamlit as st
import numpy as np
import pandas as pd
import time
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from streamlit_lottie import st_lottie


movies_data = pd.read_csv('movies.csv', low_memory=False)

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

# Save the similarity matrix to file
if not os.path.exists('trained_model.pkl'):
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(similarity, f)

# Defining function for movie recommendation
def get_movie_recommendations(input_genre='', input_name=''):
    # Finding the movies with the given genre or name
    if input_genre:
        indices = movies_data[movies_data['genres'].str.contains(input_genre, case=False)]['index'].tolist()
    elif input_name:
        indices = movies_data[movies_data['title'].str.contains(input_name, case=False)]['index'].tolist()
    else:
        return None


    similarity_scores = []
    for index in indices:
        similarity_score = list(enumerate(similarity[index]))
        similarity_scores.append(similarity_score)

    # Flatten the list of similarity scores
    similarity_scores = [score for sublist in similarity_scores for score in sublist]

    
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    
    return sorted_similar_movies[:10]

# Creating a Streamlit app
def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
st.write("<h1 style='text-align: center; font-size: 4em; font-family:Times New Roman; color: #000000;'>MOVIEVERSE</h1>", unsafe_allow_html=True)

lottie_url_hello = "https://assets6.lottiefiles.com/packages/lf20_CTaizi.json"
lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello, key="hello",height=400)

# Ask user for input genre or name

col1, col2 = st.columns(2)
input_genre = col1.text_input('Enter the movie genre (leave blank if not applicable):')
input_name = col2.text_input('Enter the movie name (leave blank if not applicable):')


if st.button('Get Recommendations'):
    
    recommendations = get_movie_recommendations(input_genre=input_genre, input_name=input_name)

    st.balloons()

    
    def display_recommendations(recommendations):
        num_cols = 2
        col_width = 450

        cols = st.columns(num_cols)

        for i, movie in enumerate(recommendations):
        # Getting the movie title from the index
            index = movie[0]
            title_from_index = movies_data[movies_data['index'] == index]['title'].values[0]

        # Calculating the column and row indices
            col_idx = i % num_cols
            row_idx = i // num_cols

        # Displaying the movie title in the appropriate column and row
            with cols[col_idx]:
                st.write('{}. {}'.format(i+1, title_from_index), width=col_width)
    

    if recommendations:
        if input_genre:
            st.write('Movies suggested for you based on your favourite genre {}:'.format(input_genre))
        elif input_name:
            st.write('Movies suggested for you based on your favourite movie {}:'.format(input_name))

        # for i, movie in enumerate(recommendations):
        #     index = movie[0]
        #     title_from_index = movies_data[movies_data['index'] == index]['title'].values[0]
        #     st.write('{}. {}'.format(i+1, title_from_index))

        display_recommendations(recommendations)



    else:
        st.write('Please enter either a genre or a movie name to get recommendations.')

