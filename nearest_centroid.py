import pandas as pd
import numpy as np
import string
import re

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

def pred4a(text,df):
    train_a = df['content'].tolist()
    vectorizer = CountVectorizer(stop_words="english", tokenizer=word_tokenize, ngram_range=(1,1),max_features=10000, analyzer="word")
    vectorizer_train_a = vectorizer.fit_transform(train_a)
    vec=vectorizer.transform([clean(text)]).toarray()
    ltag=df.tags.tolist()
    ltag1 = [item for sublist in ltag for item in sublist]
    s = pd.Series([ str(ltag[x]).strip('[]') for x in range(len(ltag))])
    dict_sim={}
    keys=list(set(ltag1))
    for i in keys:
        tg1=train1[s.str.contains(i,case=False)]
        ind1=tg1.index.values
        X=vectorizer_train_a[ind1].toarray()
        kmeans_model = KMeans(n_clusters=1).fit(X)
        dict_sim[i]=cosine_similarity(vec,np.array(kmeans_model.cluster_centers_))[0,0]
    dg={ key:value for key, value in dict_sim.items() if value >= 0.1}
    if len(dg)!=0:
        if len(dg)>=3:
            p1=dict((x, y) for x, y in sorted(dg.items(), key=lambda x:x[1],reverse=True)[0:3])
        else:
            p1=dict((x, y) for x, y in sorted(dg.items(), key=lambda x:x[1],reverse=True)[0:len(dg)])
    else:
        dg2 = { key:value for key, value in dict_sim.items() if value < 0.1}
        p1=dict((x, y) for x, y in sorted(dg2.items(), key=lambda x:x[1],reverse=True)[0:1])
    return p1.keys()

def testmet4a(df):
    lx=list(df['content'])
    nl=len(lx)
    lp=[]
    for i in lx:
        lp.append(pred4a(i,train1))
    return lp

df=pd.read_json('export-2018-02-09.json')
train,test=train_test_split(df,test_size=.2)
train1=train.reset_index().drop('index', axis=1)
print(testmet4a(df))
