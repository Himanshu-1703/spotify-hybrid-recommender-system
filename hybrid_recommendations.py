import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity



class HybridRecommenderSystem:
    
    def __init__(self,  
                 number_of_recommendations: int, 
                 weight_content_based: float):
        
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based
        
        
    def __calculate_content_based_similarities(self,song_name, artist_name, songs_data,transformed_matrix):
        # filter out the song from data
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        # get the index of song
        song_index = song_row.index[0]
        # generate the input vector
        input_vector = transformed_matrix[song_index].reshape(1,-1)
        # calculate similarity scores
        content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)
        return content_similarity_scores
        
    
    def __calculate_collaborative_filtering_similarities(self, song_name, artist_name, track_ids, songs_data, interaction_matrix):
        # fetch the row from songs data
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        # track_id of input song
        input_track_id = song_row['track_id'].values.item()
        # index value of track_id
        ind = np.where(track_ids == input_track_id)[0].item()
        # fetch the input vector
        input_array = interaction_matrix[ind]
        # get similarity scores
        collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)
        return collaborative_similarity_scores
    
    
    def __normalize_similarities(self, similarity_scores):
        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
        return normalized_scores
    
    
    def __weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        weighted_scores = (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_filtering_scores)
        return weighted_scores
    
    
    def give_recommendations(self, song_name, artist_name, songs_data, track_ids, transformed_matrix, interaction_matrix):
        # calculate content based similarities
        content_based_similarities = self.__calculate_content_based_similarities(song_name= song_name, 
                                                                               artist_name= artist_name, 
                                                                               songs_data= songs_data, 
                                                                               transformed_matrix= transformed_matrix)
        
        # calculate collaborative filtering similarities
        collaborative_filtering_similarities = self.__calculate_collaborative_filtering_similarities(song_name= song_name, 
                                                                                                   artist_name= artist_name, 
                                                                                                   track_ids= track_ids, 
                                                                                                   songs_data= songs_data, 
                                                                                                   interaction_matrix= interaction_matrix)
    
        # normalize content based similarities
        normalized_content_based_similarities = self.__normalize_similarities(content_based_similarities)
        
        # normalize collaborative filtering similarities
        normalized_collaborative_filtering_similarities = self.__normalize_similarities(collaborative_filtering_similarities)
        
        # weighted combination of similarities
        weighted_scores = self.__weighted_combination(content_based_scores= normalized_content_based_similarities, 
                                                    collaborative_filtering_scores= normalized_collaborative_filtering_similarities)
        
        
        # index values of recommendations
        recommendation_indices = np.argsort(weighted_scores.ravel())[-self.number_of_recommendations-1:][::-1] 
        
        # get top k recommendations
        recommendation_track_ids = track_ids[recommendation_indices]
       
        # get top scores
        top_scores = np.sort(weighted_scores.ravel())[-self.number_of_recommendations-1:][::-1]
        
        # get the songs from data and print
        scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),
                                "score":top_scores})
        top_k_songs = (
                        songs_data
                        .loc[songs_data["track_id"].isin(recommendation_track_ids)]
                        .merge(scores_df,on="track_id")
                        .sort_values(by="score",ascending=False)
                        .drop(columns=["track_id","score"])
                        .reset_index(drop=True)
                        )
        
        return top_k_songs
    
    