import pandas as pd
from data_cleaning import data_for_content_filtering
from content_based_filtering import transform_data, save_transformed_data

# path of filtered data
filtered_data_path = "data/collab_filtered_data.csv"

# save path
save_path = "data/transformed_hybrid_data.npz"


def main(data_path, save_path):
    # load the filtered data
    filtered_data = pd.read_csv(data_path)

    # clean the data
    filtered_data_cleaned = data_for_content_filtering(filtered_data)

    # transform the data into matrix
    transformed_data = transform_data(filtered_data_cleaned)

    # save the transformed data
    save_transformed_data(transformed_data, save_path)
    

if __name__ == "__main__":
    main(filtered_data_path, save_path)