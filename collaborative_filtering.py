import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# set paths
# output paths
track_ids_save_path = "data/track_ids.npy"
filtered_data_save_path = "data/collab_filtered_data.csv"
interaction_matrix_save_path = "data/interaction_matrix.npz"
# input paths
songs_data_path = "data/cleaned_data.csv"
user_listening_history_data_path = "data/User Listening History.csv"


def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
    """
    Filter the songs data for the given track ids
    """
    # filter data based on track_ids
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)]
    # rest index
    filtered_data.reset_index(drop=True, inplace=True)
    # save the data
    save_pandas_data_to_csv(filtered_data, save_df_path)
    
    return filtered_data


def save_pandas_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the data to a csv file
    """
    data.to_csv(file_path, index=False)
    
    
def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    """
    Save the sparse matrix to a npz file
    """
    save_npz(file_path, matrix)


def create_interaction_matrix(history_data:dd.DataFrame, track_ids_save_path, save_matrix_path) -> csr_matrix:
    # make a copy of data
    df = history_data.copy()
    
    # convert the playcount column to float
    df['playcount'] = df['playcount'].astype(np.float64)
    
    # convert string column to categorical
    df = df.categorize(columns=['user_id', 'track_id'])
    
    # Convert user_id and track_id to numeric indices
    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes
    
    # get the list of track_ids
    track_ids = df['track_id'].cat.categories.values
    
    # save the categories
    np.save(track_ids_save_path, track_ids, allow_pickle=True)
    
    # add the index columns to the dataframe
    df = df.assign(
        user_idx=user_mapping,
        track_idx=track_mapping
    )
    
    # create the interaction matrix
    interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
    
    # compute the matrix
    interaction_matrix = interaction_matrix.compute()
    
    # get the indices to form sparse matrix
    row_indices = interaction_matrix['track_idx']
    col_indices = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']
    
    # get the shape of sparse matrix
    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()
    
    # create the sparse matrix
    interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    
    # save the sparse matrix
    save_sparse_matrix(interaction_matrix, save_matrix_path)
    
    
def collaborative_recommendation(song_name,artist_name,track_ids,songs_data,interaction_matrix,k=5):
    # lowercase the song name
    song_name = song_name.lower()
    
    # lowercase the artist name
    artist_name = artist_name.lower()
    
    # fetch the row from songs data
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
   
    # track_id of input song
    input_track_id = song_row['track_id'].values.item()
  
    # index value of track_id
    ind = np.where(track_ids == input_track_id)[0].item()
    
    # fetch the input vector
    input_array = interaction_matrix[ind]
    
    # get similarity scores
    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    
    # index values of recommendations
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    
    # get top k recommendations
    recommendation_track_ids = track_ids[recommendation_indices]
    
    # get top scores
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    
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


def main():
    # load the history data
    user_data = dd.read_csv(user_listening_history_data_path)
    
    # get the unique track ids
    unique_track_ids = user_data.loc[:,"track_id"].unique().compute()
    unique_track_ids = unique_track_ids.tolist()
    
    # filter the songs data
    songs_data = pd.read_csv(songs_data_path)
    filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)
    
    # create the interaction matrix
    create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)


if __name__ == "__main__":
    main()