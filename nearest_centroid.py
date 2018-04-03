import pandas as pd
import numpy as np
import string
import re
import time

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

#English Stop Words
stop=set(stopwords.words('english'))
#Punctuation 
exclude=set(string.punctuation)
# Replace functional approach
repls=("(",""),(")",""),("'s","")
#Exclusion Words
#words_ex=['room','mr','please','mrs','guest']
words_ex=[]
#Lemmatization
lemma=WordNetLemmatizer()

#Function to clean text

def clean(text):
    #Remove specific chars
    rem1=re.sub("[!#*%:1234567890?,.;&-]","",text)
    #Remove non ASCII chars
    rem2=re.sub(r'[^\x00-\x7f]',r' ',rem1)
    #Replace using functional approach
    rem3=reduce(lambda a,kv:a.replace(*kv),repls,rem2)   
    #Exclude stopwords
    stop_free=" ".join([i for i in rem2.lower().split()
                        if i not in stop])
    #Exclude Punctuation
    punc_free=''.join(i for i in stop_free if i not in exclude)
    #Exclude words
    cust_words=" ".join(i for i in punc_free.split()
                        if i not in words_ex)
    #Exclude numbers
    num_free=''.join([i for i in cust_words if not i.isdigit()])
    #Lemmatization
    normalized=" ".join(lemma.lemmatize(word)
                        for word in num_free.split())
    return normalized

def nearest_centroid(data):
    train,test=train_test_split(df,test_size=.2)
    train=train.reset_index().drop('index',axis=1)
    texts=list(data['content'])
    train_texts=train['content'].tolist()
    vectorizer=CountVectorizer(stop_words='english',
        tokenizer=word_tokenize,ngram_range=(1,1),max_features=10000,
        analyzer='word')
    train_vectorizer=vectorizer.fit_transform(train_texts)
    tags=list(set([item for sublist in train.tags.tolist()
                   for item in sublist]))
    s=pd.Series([str(train.tags.tolist()[x]).strip('[]')
                 for x in range(len(train.tags.tolist()))])
    predictions=[]
    for text in texts:
        vec=vectorizer.transform([clean(text)]).toarray() 
        dict_sim={}
        for tag in tags:
            tg1=train[s.str.contains(tag,case=False)]
            ind1=tg1.index.values
            X=train_vectorizer[ind1].toarray()
            kmeans_model=KMeans(n_clusters=1).fit(X)
            dict_sim[tag]=cosine_similarity(vec,np.array(
                kmeans_model.cluster_centers_))[0,0]
        dg={key:value for key,value in dict_sim.items() if value>=0.1}
        if len(dg)>=3:p1=dict((x,y) for x,y in
            sorted(dg.items(),key=lambda x:x[1],reverse=True)[0:3])
        elif len(dg)>0:p1=dict((x,y) for x,y in
            sorted(dg.items(),key=lambda x:x[1],reverse=True)[0:len(dg)])
        else:p1=dict((x, y) for x, y in
            sorted(dict_sim.items(), key=lambda x:x[1],
                   reverse=True)[0:1])
        predictions.append(p1.keys())
    predictions=pd.DataFrame({'predictions':predictions})
    predictions['original']=data.tags
    return predictions

df=pd.read_json('export-2018-02-09.json')
train,test=train_test_split(df,test_size=.2)
train1=train.reset_index().drop('index', axis=1)
start=time.time()
print(nearest_centroid(df))
end=time.time()
print('time: '+str(end-start))
