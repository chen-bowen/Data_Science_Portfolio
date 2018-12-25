import json
import plotly
import pandas as pd
import re
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
        Tokenize the message into word level features. 
        1. replace urls
        2. convert to lower cases
        3. remove stopwords
        4. strip white spaces
    Args: 
        text: input text messages
    Returns: 
        cleaned tokens(List)
    """   
    # Define url pattern
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect and replace urls
    detected_urls = re.findall(url_re, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize sentences
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # save cleaned tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    # remove stopwords
    STOPWORDS = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]
    
    return clean_tokens


# load data

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/rf_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visuals
    # Message counts of different generes
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Message counts for different categories
    cate_counts_df = df.iloc[:, 4:].sum().sort_values(ascending=False)
    cate_counts = list(cate_counts_df)
    cate_names = list(cate_counts_df.index)

    # Top keywords in Social Media in percentages
    social_media_messages = ' '.join(df[df['genre'] == 'social']['message'])
    social_media_tokens = tokenize(social_media_messages)
    social_media_wrd_counter = Counter(social_media_tokens).most_common()
    social_media_wrd_cnt = [i[1] for i in social_media_wrd_counter]
    social_media_wrd_pct = [i/sum(social_media_wrd_cnt) *100 for i in social_media_wrd_cnt]
    social_media_wrds = [i[0] for i in social_media_wrd_counter]

    # Top keywords in Direct in percentages
    direct_messages = ' '.join(df[df['genre'] == 'direct']['message'])
    direct_tokens = tokenize(direct_messages)
    direct_wrd_counter = Counter(direct_tokens).most_common()
    direct_wrd_cnt = [i[1] for i in direct_wrd_counter]
    direct_wrd_pct = [i/sum(direct_wrd_cnt) * 100 for i in direct_wrd_cnt]
    direct_wrds = [i[0] for i in direct_wrd_counter]

    # create visuals

    graphs = [
    # Histogram of the message genere
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # histogram of social media messages top 30 keywords 
        {
            'data': [
                    Bar(
                        x=social_media_wrds[:50],
                        y=social_media_wrd_pct[:50]
                                    )
            ],

            'layout':{
                'title': "Top 50 Keywords in Social Media Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "% Total Social Media Messages"    
                }
            }
        }, 

        # histogram of direct messages top 30 keywords 
        {
            'data': [
                    Bar(
                        x=direct_wrds[:50],
                        y=direct_wrd_pct[:50]
                                    )
            ],

            'layout':{
                'title': "Top 50 Keywords in Direct Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "% Total Direct Messages"    
                }
            }
        }, 



        # histogram of messages categories distributions
        {
            'data': [
                    Bar(
                        x=cate_names,
                        y=cate_counts
                                    )
            ],

            'layout':{
                'title': "Distribution of Message Categories",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "count"    
                }
            }
        },     

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()