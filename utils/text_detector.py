from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

def detect_fake_news(text):
    
    # yaha tumhara ML model use hoga

    label = "FAKE"
    score = 0.87

    return label, score