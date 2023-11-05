import os
from flask import Flask, render_template, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__, static_url_path='/static')

sia = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html', text=None, sentiment=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        print(text)
        sentiment = get_sentiment(text)
        return jsonify({'text': text, 'sentiment': sentiment})

def get_sentiment(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

if __name__ == '__main__':
    app.run(debug=True)
