import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from collaborative_filtering import collaborative_recommendation
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem


# load the data
cleaned_data_path = "data/cleaned_data.csv"
st.session_state.songs_data = pd.read_csv(cleaned_data_path)

# load the transformed data
transformed_data_path = "data/transformed_data.npz"
st.session_state.transformed_data = load_npz(transformed_data_path)

# load the track ids
track_ids_path = "data/track_ids.npy"
st.session_state.track_ids = load(track_ids_path,allow_pickle=True)

# load the filtered songs data
filtered_data_path = "data/collab_filtered_data.csv"
st.session_state.filtered_data = pd.read_csv(filtered_data_path)

# load the interaction matrix
interaction_matrix_path = "data/interaction_matrix.npz"
st.session_state.interaction_matrix = load_npz(interaction_matrix_path)

# load the transformed hybrid data
transformed_hybrid_data_path = "data/transformed_hybrid_data.npz"
st.session_state.transformed_hybrid_data = load_npz(transformed_hybrid_data_path)

# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text Input
song_name = st.text_input('Enter a song name:')
st.write('You entered:', song_name)
# artist name
artist_name = st.text_input('Enter the artist name:')
st.write('You entered:', artist_name)
# lowercase the input
song_name = song_name.lower()
artist_name = artist_name.lower()

# k recommndations
k = st.selectbox('How many recommendations do you want?', [5,10,15,20], index=1)

if ((st.session_state.filtered_data["name"] == song_name) & (st.session_state.filtered_data["artist"] == artist_name)).any():   
    # type of filtering
    filtering_type = st.selectbox(label= 'Select the type of filtering:', 
                                options= ['Content-Based Filtering', 
                                            'Collaborative Filtering',
                                            "Hybrid Recommender System"],
                                index= 2)

    # diversity slider
    diversity = st.slider(label="Diversity in Recommendations",
                        min_value=1,
                        max_value=10,
                        value=5,
                        step=1)

    content_based_weight = 1 - (diversity / 10)
else:
    # type of filtering
    filtering_type = st.selectbox(label= 'Select the type of filtering:', 
                                options= ['Content-Based Filtering'])

# Button
if filtering_type == 'Content-Based Filtering':
    if st.button('Get Recommendations'):
        if ((st.session_state.songs_data["name"] == song_name) & (st.session_state.songs_data['artist'] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = content_recommendation(song_name=song_name,
                                                     artist_name=artist_name,
                                                     songs_data=st.session_state.songs_data,
                                                     transformed_data=st.session_state.transformed_data,
                                                     k=k)
            
            # Display Recommendations
            for ind , recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()
                
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")
            
elif filtering_type == 'Collaborative Filtering':
    if st.button('Get Recommendations'):
        if ((st.session_state.filtered_data["name"] == song_name) & (st.session_state.filtered_data["artist"] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = collaborative_recommendation(song_name=song_name,
                                                           artist_name=artist_name,
                                                           track_ids=st.session_state.track_ids,
                                                           songs_data=st.session_state.filtered_data,
                                                           interaction_matrix=st.session_state.interaction_matrix,
                                                           k=k)
            
            # Display Recommendations
            for ind , recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()
                
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")

elif filtering_type == "Hybrid Recommender System":
    if st.button('Get Recommendations'):
        if ((st.session_state.filtered_data["name"] == song_name) & (st.session_state.filtered_data["artist"] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommender = HybridRecommenderSystem(
                                                    number_of_recommendations= k,
                                                    weight_content_based= content_based_weight
                                                    )
                                    
            # get the recommendations
            recommendations = recommender.give_recommendations(song_name= song_name,
                                                            artist_name= artist_name,
                                                            songs_data= st.session_state.filtered_data,
                                                            transformed_matrix= st.session_state.transformed_hybrid_data,
                                                            track_ids= st.session_state.track_ids,
                                                            interaction_matrix= st.session_state.interaction_matrix)
            # Display Recommendations
            for ind , recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()
                
                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")