import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz

# Cleaned Data Path
CLEANED_DATA_PATH = "data/cleaned_data.csv"

# cols to transform
frequency_enode_cols = ['year']
ohe_cols = ['artist',"time_signature","key"]
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms","loudness","tempo"]
min_max_scale_cols = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]


def train_transformer(data):
    # transformer 
    transformer = ColumnTransformer(transformers=[
        ("frequency_encode", CountEncoder(normalize=True,return_df=True), frequency_enode_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
        ("standard_scale", StandardScaler(), standard_scale_cols),
        ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
    ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)

    # fit the transformer
    transformer.fit(data)

    # save the transformer
    joblib.dump(transformer, "transformer.joblib")
    

def transform_data(data):
    # load the transformer
    transformer = joblib.load("transformer.joblib")
    
    # transform the data
    transformed_data = transformer.transform(data)
    
    return transformed_data


def save_transformed_data(transformed_data,save_path):
    # save the transformed data
    save_npz(save_path, transformed_data)


def calculate_similarity_scores(input_vector, data):
    # calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, data)
    
    return similarity_scores


def recommend(song_name, songs_data, transformed_data, k=10):
    # convert song name to lowercase
    song_name = song_name.lower()
    # filter out the song from data
    song_row = songs_data.loc[songs_data["name"] == song_name]
    # get the index of song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)
    # calculate similarity scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    # print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    return top_k_list

def main(data_path, song_name, k=10):
    # convert song name to lowercase
    song_name = song_name.lower()
    # load the data
    data = pd.read_csv(data_path)
    # clean the data
    data_content_filtering = data_for_content_filtering(data)
    # train the transformer
    train_transformer(data_content_filtering)
    # transform the data
    transformed_data = transform_data(data_content_filtering)
    #save transformed data
    save_transformed_data(transformed_data,"data/transformed_data.npz")
    # filter out the song from data
    song_row = data.loc[data["name"] == song_name]
    # get the index of song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)
    # calculate similarity scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:-1][::-1]
    # get the top k songs names
    top_k_songs_names = data.iloc[top_k_songs_indexes]
    # print the top k songs
    print(top_k_songs_names)
    
if __name__ == "__main__":
    main(CLEANED_DATA_PATH, "Hips Don't Lie")