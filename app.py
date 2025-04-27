from flask import Flask, render_template, request
from apify_client import ApifyClient
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

# Load model
model = load_model('insta-fake-real.h5')

# Apify Token
APIFY_TOKEN = "apify_api_WXRA1o27OqXniZ02Za4561iNfc4U7Q01M0fP"

# Feature columns
columns = [
    "profile_pic", "username_length", "fullname_words", "fullname_length", 
    "name_equals_username", "description_length", "external_url", "is_private", 
    "posts_count", "followers_count", "following_count"
]

# Define scaler
scaler = MinMaxScaler()
scaler.fit([
    [3, 0],    # min values
    [30, 50]   # max values
])

def get_insta_profile_data(username):
    client = ApifyClient(APIFY_TOKEN)

    run_input = {
        "usernames": [username],
        "resultsLimit": 1,
    }

    run = client.actor("apify/instagram-profile-scraper").call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

    if not items:
        return None

    profile = items[0]
    username_val = profile.get("username", "")
    full_name_val = profile.get("fullName", "")
    description_val = profile.get("bio", "")
    profile_pic_url = profile.get("profilePicUrl", "")

    external_url = bool(profile.get("externalUrl", ""))
    is_private = profile.get("private", False)
    posts_count = profile.get("postsCount", 0)
    followers_count = profile.get("followersCount", 0)
    following_count = profile.get("followsCount", 0)

    raw_lengths = np.array([[len(username_val), len(full_name_val)]])
    scaled_lengths = scaler.transform(raw_lengths)[0]

    data = {
        "profile_pic": 1 if profile_pic_url else 0,
        "username_length": scaled_lengths[0],
        "fullname_words": len(full_name_val.split()),
        "fullname_length": scaled_lengths[1],
        "name_equals_username": 1 if username_val == full_name_val else 0,
        "description_length": len(description_val),
        "external_url": 1 if external_url else 0,
        "is_private": 1 if is_private else 0,
        "posts_count": posts_count,
        "followers_count": followers_count,
        "following_count": following_count,
        "profile_pic_url": profile_pic_url
    }

    return data

def predict_real_or_fake(profile_data):
    df = pd.DataFrame([profile_data])[columns]
    input_data = np.array(df).reshape(1, -1)
    prediction = model.predict(input_data)[0][0]
    return prediction

def get_image_base64(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    profile_data = None
    profile_img_base64 = None
    username = None

    if request.method == 'POST':
        username = request.form['username']

        profile_data = get_insta_profile_data(username)
        if profile_data:
            prediction = predict_real_or_fake(profile_data)

            if prediction >= 0.5:
                prediction_result = f"✅ This is likely a REAL account. (Confidence: {prediction:.2f})"
            else:
                prediction_result = f"⚠️ This is likely a FAKE account. (Confidence: {1 - prediction:.2f})"

            # Fetch profile picture
            if profile_data["profile_pic_url"]:
                profile_img_base64 = get_image_base64(profile_data["profile_pic_url"])

    return render_template('index.html', 
                           username=username,
                           prediction_result=prediction_result,
                           profile_data=profile_data,
                           profile_img_base64=profile_img_base64)

if __name__ == '__main__':
    app.run(debug=True)
