import requests
import nltk
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

nlp = spacy.load("en_core_web_sm")
# import the html file from the news article
r = requests.get(
    "https://www.newsy.com/stories/commercial-companies-advance-space-exploration/"
)
r.encoding = "utf-8"
html = r.text

# print(html[0:500])

# we used beautiful soup to extract text from the html file
soup = BeautifulSoup(html, features="lxml")

text = soup.get_text()

# print(len(text))
# print(text[100:1100])

# cleaned the text and removed unecessary stuff
clean_text = text.replace("/n", " ")
clean_text = clean_text.replace("_", " ")
clean_text = clean_text.replace("/", " ")

clean_text = "".join([c for c in clean_text if c != "'"])


# print(clean_text)
"""
 dividing into sentences
 nlp is natural language processing, which is being done using spaCy lib

"""

sentence = []
tokens = nlp(clean_text)

# token.sents will look at the the text and (sents) represents sentences here
for sent in tokens.sents:
    sentence.append((sent.text.strip()))  # add it to the list


"""
Using BERT for sentiment analysis 

reference article: https://medium.com/mlearning-ai/tweets-sentiment-analysis-with-roberta-1f30cf4e1035

"""

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)

tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ["Negative", "Neutral", "Positive"]

for lines in sentence:
    encoded_tweet = tokenizer(lines, return_tensors="pt")

    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    for i in range(len(scores)):
        # we can get all the probabilities which can be later used to derive the overall sentiment of the article
        l = labels[i]
        s = scores[i]
        print(l, s)
