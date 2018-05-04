import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk import word_tokenize

data=pd.read_json('export-2018-02-09.json')

params={"n_estimators":170,"max_depth":5,"random_state":10,"min_samples_split":4,"min_samples_leaf":2}
classifier=OneVsRestClassifier(GradientBoostingClassifier(**params))
mlb=MultiLabelBinarizer(classes=list(set([item for sublist in data.tags.tolist() for item in sublist])))

X_raw=data['content'].as_matrix()
y_raw=data['tags'].as_matrix()
vectorizer=TfidfVectorizer(stop_words='english',tokenizer=word_tokenize,ngram_range=(1,3),max_features=10000,analyzer='word')
X=vectorizer.fit_transform(X_raw)
y=mlb.fit_transform(y_raw)

scores=cross_val_score(classifier,X,y,cv=10)
