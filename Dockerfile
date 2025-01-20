# set up the base image
FROM python:3.12

# set the working directory
WORKDIR /app/

# copy the requirements file to workdir
COPY requirements.txt .

# install the requirements
RUN pip install -r requirements.txt

# copy the data files and code files
COPY ./data/collab_filtered_data.csv ./data/collab_filtered_data.csv
COPY ./data/interaction_matrix.npz ./data/interaction_matrix.npz
COPY ./data/track_ids.npy ./data/track_ids.npy
COPY ./data/cleaned_data.csv ./data/cleaned_data.csv
COPY ./data/transformed_data.npz ./data/transformed_data.npz
COPY ./data/transformed_hybrid_data.npz ./data/transformed_hybrid_data.npz
COPY app.py app.py
COPY collaborative_filtering.py collaborative_filtering.py
COPY content_based_filtering.py content_based_filtering.py
COPY hybrid_recommendations.py hybrid_recommendations.py
COPY data_cleaning.py data_cleaning.py
COPY transform_filtered_data.py transform_filtered_data.py

# expose the port on the container
EXPOSE 8000

# run the streamlit app
CMD [ "streamlit", "run", "app.py", "--server.port" ,"8000" ]