from flask import Flask, render_template, request
import os
from pytube import extract
import googleapiclient.discovery
import pandas as pd 
import pickle
import re
import string
from dotenv import load_dotenv

import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

load_dotenv()

import mysql.connector

# ytcdb = mysql.connector.connect(
#     host='localhost',
#     user='aryanmain',
#     password='aryan123',
#     db='ytcomments',
    
# )

ytcdb = mysql.connector.connect(
    
    host = os.getenv('DB_HOST'),
    user = os.getenv('DB_USER'),
    password = os.getenv('DB_PASSWORD'),
    database = os.getenv('DB_NAME')
    
)




mycursor = ytcdb.cursor()


STOPWORDS=set(stopwords.words('english'))



app = Flask(__name__)

with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('vectoriser_text.pkl', 'rb') as f:
    vectoriser = pickle.load(f)
    
    

def google_api(id):
    # os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = os.getenv("YOUR_API_KEY")

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        maxResults=100,
        order="relevance",
        videoId= id
    )
    response = request.execute()

    return response

def clean_text(df_column):
    cleaned_texts = []
    for text in df_column:
        cleaned_text = ''
        text = ' '.join([word.lower() for word in text.split()]) 
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\S+\s?', '', text)
        text = re.sub('[0-9]+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"(.)\1\1+", r"\1\1", text)

        for i in text:
            if i.isalnum(): 
                cleaned_text += i
            else:
                cleaned_text += " "

        text_t = word_tokenize(cleaned_text)
        filtered_text = [word for word in text_t if word not in STOPWORDS]

        ps = PorterStemmer()
        stemed_words = [ps.stem(word) for word in filtered_text ]

        lm = WordNetLemmatizer()
        lemm_words = [lm.lemmatize(word,pos='a') for word in stemed_words]

        cleaned_texts.append(' '.join(lemm_words))
        
    return cleaned_texts


def parseurl(url):
    id=extract.video_id(url)
    return id 
    

def create_df_author_comments(response):
  authorname = []
  comments = []
  for i in range(len(response["items"])):
    authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
    comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
  df = pd.DataFrame({'author_name': authorname, 'Comments': comments})
  return df


def res_sep(list):
    res = []
    for i in range(len(list)):
        res.append(list[i])
    return res

def classify(list):
  results = []
  for i in range(len(list)):
    results.append(classifier.predict(vectoriser.transform(clean_text([list[i]]))[0]))
  result = res_sep(results)
  return result



    

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Get the URL of the YouTube video from the form
    video_url = request.form["video_url"]
    video_id = parseurl(video_url)
    
    # Retrieve the comments using YouTube API
    response = google_api(video_id)
    
    # Create a DataFrame from the comments
    comments_author = create_df_author_comments(response)
    
    
    
    # Classify the comments using the pre-trained model
    results = pd.DataFrame(res_sep(classify(comments_author["Comments"])),columns=["sentiment"])
  


    df = pd.merge(comments_author.reset_index(drop=True), results.reset_index(drop=True), how='outer', left_index=True, right_index=True)
    # print(df.shape)
    sql = "INSERT INTO ytca (author_name,comments,sentiment) VALUES (%s, %s, %s)"
    values = df.values.tolist()

    mycursor.executemany(sql,values)
    print(mycursor.rowcount, "record(s) inserted.")


    ytcdb.commit()


   

    
    # Calculate the percentage of positive and negative comments
    positive = round((results['sentiment'] == 1).sum()/len(results) * 100)
    negative = round((results['sentiment'] == 0).sum()/len(results) * 100)

    
    # Display the result
    return f"{positive:.1f}% of comments are Positive and {negative:.1f}% are Negative <br><br><a href='/'>Analyze another video{df}</a>"
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)


