import requests
import time

app_url = "http://localhost:8000"

# get the status code
def get_app_status(url):
    response = requests.get(url)
    status_code = response.status_code
    return status_code

# test for the app home page loading
def test_app_loading():
    # wait for the app to load
    time.sleep(60)
    status_code = get_app_status(app_url)
    assert status_code == 200, "Unable to load Streamlit App"
    print("Streamlit App Loaded Successfully")