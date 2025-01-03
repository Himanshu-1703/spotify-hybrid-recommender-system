import pandas as pd


DATA_PATH = "data/Music Info.csv"

def clean_data(data):
    return (
        data
        .drop_duplicates(subset="spotify_id")
        .drop(columns=["genre","spotify_id"])
        .fillna({"tags":"no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower()
        )
    )
    
    
def data_for_content_filtering(data):
    return (
        data
        .drop(columns=["track_id","name","spotify_preview_url"])
    )
    
    
def main(data_path):
    # load the data
    data = pd.read_csv(data_path)
    
    # perform data cleaning
    cleaned_data = clean_data(data)
    
    # saved cleaned data
    cleaned_data.to_csv("data/cleaned_data.csv",index=False)
    

if __name__ == "__main__":
    main(DATA_PATH)