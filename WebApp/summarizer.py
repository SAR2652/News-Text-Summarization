import re, heapq
from flask import Flask, render_template, request
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

app = Flask("summarizer")

def extractive_summarize(news):
    # Text preprocessing
    text = re.sub(r'\[[0-9]*\]', ' ', news)  # remove numeric citations
    text = re.sub(r'\s+', ' ', text)  # remove 1+ continuous whitespaces
    clean_text = text.lower()  # convert all text to lower case
    clean_text = re.sub(r'\W', ' ', clean_text)  # remove non-word characters
    # clean_text = re.sub(r'\d', ' ', clean_text)  # remove any digits
    # remove 1+ continuous whitespaces
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # Tokenization
    sentences = sent_tokenize(text)
    ratio = 0.4
    summary_length = int(ratio * len(sentences))
    stop_words = stopwords.words('english')

    # Initialize an empty dictionary to store the count of words
    word2count = {}
    for word in word_tokenize(clean_text):
        if word not in stop_words:
            if word not in word2count.keys():  # create new key as word
                word2count[word] = 1
            else:
                word2count[word] += 1

    # normalize all values
    for key in word2count.keys():
        word2count[key] = word2count[key] / max(word2count.values())

    # Generate a dictionary of sentence scores
    sent2score = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word2count.keys():
                if len(sentence.split(' ')) < 30:   # sentence length condition
                    if sentence not in sent2score.keys():
                        sent2score[sentence] = word2count[word]
                    else:
                        sent2score[sentence] += word2count[word]

    best_sentences = heapq.nlargest(summary_length, sent2score, key=sent2score.get)
    return '.\n'.join(best_sentences)

@app.route("/")
def formpage():
    return render_template("form.html")

@app.route("/get_summary", methods = ['POST'])
def get_summary():
    news = request.form['news']
    summary = extractive_summarize(news)
    return render_template('result.html', summary = summary)
