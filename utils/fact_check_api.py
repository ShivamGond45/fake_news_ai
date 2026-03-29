import requests

API_KEY = "AIzaSyBItyJzoEolUtG9ISZnfuUIQyNZyY9Jc4M"

def fact_check_news(text):

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    params = {
        "query": text,
        "key": API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if "claims" in data:

            claim = data["claims"][0]

            review = claim["claimReview"][0]

            publisher = review["publisher"]["name"]
            rating = review["textualRating"]

            return f"⚠️ Fact Check Found: {rating} (Source: {publisher})"

        else:
            return "✅ No fact-check found (May be real)"

    except:
        return "❌ API Error"