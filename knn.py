import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from tagging.helper import clean

def knn(data,k,include_books=True):
    texts=list(data['content'])
    tags=data.tags.tolist()
    predictions=[]
    vectorizer=TfidfVectorizer(stop_words='english',tokenizer=word_tokenize,
                               ngram_range=(1,3),max_features=10000,analyzer='word')
    X=vectorizer.fit_transform(data['content'].as_matrix())
    if include_books:
        books=list(data['book_name'].apply(lambda x:x[0]).as_matrix())
        vectorizer_b=TfidfVectorizer(stop_words='english',tokenizer=word_tokenize,
                                      ngram_range=(1,3),max_features=10000,analyzer='word')
        X_b=vectorizer_b.fit_transform(data['book_name'].apply(lambda x:x[0]).as_matrix())
        X=hstack([X_b,X]).toarray()
        for text,books in zip(texts,books):
            dist_cos={}
            lcs=[]
            vec_t=vectorizer.transform([clean(text)])
            vec_b=vectorizer_b.transform([clean(books)])
            vec=hstack([vec_b,vec_t]).toarray()
            for i in range(data.shape[0]):
                dist_cos[i]={'tags':tags[i],'cs':cosine_similarity(vec,X[i:i+1])[0,0]}
                lcs.append(cosine_similarity(vec,X[i:i+1])[0,0])
            r=np.array(lcs)
            rt=[tags[j] for j in (r.argsort()[-k:][::-1]).tolist()]
            if k>1:
                a1=set(rt[1])
                for i in range(1,k):
                    a2=a1&set(rt[i])
                    a1=a2
                res=set(rt[0]) | a1
            else:
                res=set(rt[0])
            predictions.append(list(res))
    else:
        for text in texts:
            dist_cos={}
            lcs=[]
            vec=vectorizer.transform([clean(text)])
            for i in range(data.shape[0]):
                dist_cos[i]={'tags':tags[i],'cs':cosine_similarity(vec,X[i:i+1])[0,0]}
                lcs.append(cosine_similarity(vec,X[i:i+1])[0,0])
            r=np.array(lcs)
            rt=[tags[j] for j in (r.argsort()[-k:][::-1]).tolist()]
            if k>1:
                a1=set(rt[1])
                for i in range(1,k):
                    a2=a1&set(rt[i])
                    a1=a2
                res=set(rt[0]) | a1
            else:
                res=set(rt[0])
            predictions.append(list(res))
    predictions=pd.DataFrame({'predictions':predictions})
    predictions['original']=data.tags
    return predictions
