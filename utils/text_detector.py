from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

def detect_fake_news(text):

    result = classifier(text)[0]   # model ka output

    label = result['label']        # FAKE / REAL
    score = result['score']        # confidence

    return label, score