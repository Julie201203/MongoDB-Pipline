import pandas as pd
import re
import pymongo
import certifi

#Fetch MongoDB connection string
import json
with open (r'credentials.json') as f:
    data = json.load(f)
    mongo_conn_str = data['mongodb']

import pandas as pd
from datetime import datetime

client = pymongo.MongoClient(mongo_conn_str, tlsCAFile=certifi.where())

#Find data through imdb in 2010
db = client['DA320']
imdb_view = db['imdb']


# Retrieve the data from the view
data = list(imdb_view.find({"release_date": {"$regex": "2010"}}))  # Fetch 
all documents from the view

# Convert the data to a DataFrame
imdb_view = pd.DataFrame(data)


#Find data in Metacritic in the same year
metacritic_view = pd.DataFrame(db.metacritic.find({"release_date": 
re.compile("2010")}))

#Convert value to number
metacritic_view.score = pd.to_numeric(metacritic_view.score, errors = 
'coerce')
metacritic_view.release_date = 
pd.to_datetime(metacritic_view.release_date, errors = 'coerce')

#Print out 2 tables
print(imdb_view)
print(metacritic_view)

#Merging 2 data sets together
unified_view = pd.merge(imdb_view, metacritic_view, how = "inner", 
on="title")
unified_view.head()

from transformers import pipeline

#DistillBert classifier
#Testing
classifier = pipeline("fill-mask", model="distilbert-base-uncased")
classifier("I need to eat [MASK] to gain muscle.")

def askbert(row):
    prompt = "{}\n\n{}\n\nIn one word, the movie is 
[MASK].".format(row['title'], row['description'])
    print(f"prompting", row['title'])
    return classifier(prompt)[0]['token_str']

words = unified_view.apply(askbert, axis=1, raw=False, 
result_type="expand")
print(words)

unified_view['desc'] = words

import matplotlib
import matplotlib.pyplot as plt

unified_view['desc'].value_counts().plot(kind='bar')

plt.title("Number of movies in the same category")
plt.xlabel("Description")
plt.ylabel("Number of movies")
plt.show()

def askbert(row):
    prompt = "{}\n\n{}\n\n Overall this Movie is [MASK] (one of 'Great', 
'good', 'average', 'bad', or 'terrible').".format(row['title'], 
row['description'])
    print(f"prompting", row['title'], end='\r')
    return classifier(prompt)[0]['token_str']


words = unified_view.apply(askbert, axis=1, raw=False, 
result_type="expand")
print(words)

unified_view['desc'] = words

import matplotlib
import matplotlib.pyplot as plt

unified_view['desc'].value_counts().plot(kind='bar')

plt.title("Rank the movies by how good they are")
plt.xlabel("Description")
plt.ylabel("Number of movies")
plt.show()


