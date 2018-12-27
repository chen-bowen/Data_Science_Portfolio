import json
import plotly
import pandas as pd
import re
from collections import Counter

# import NLP libraries
from tokenizer_function import Tokenizer, tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


@app.before_first_request

def load_model_data():
    global df
    global model
    # load data

    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)

    # load model
    model = joblib.load("models/adaboost_model.pkl")

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
    app.run()


if __name__ == '__main__':
    main()