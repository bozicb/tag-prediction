import pandas as pd
from nltk import word_tokenize
from tagging.helper import tag_word_frequencies

def word_frequencies(data):
    predictions=[]
    word_freq=tag_word_frequencies(data)
    tags=word_freq.keys()
    for text in list(data['content']):
        met={}
        for tag in tags:
            n=0
            for word in word_tokenize(text):
                if word in word_freq[tag].keys():n+=word_freq[tag][word]
            met[tag]=n
        dg={key:value for key,value in met.items() if value>=0.1}
        if len(dg)!=0:
            if len(dg)>=3:p1=dict((x,y) for x,y in
                                  sorted(dg.items(),key=lambda x:x[1],reverse=True)[0:3])
            else:p1=dict((x,y) for x,y in
                         sorted(dg.items(),key=lambda x:x[1],reverse=True)[0:len(dg)])
        else:
            dg2={key:value for key,value in met.items() if value<0.1}
            p1=dict((x,y) for x,y in sorted(dg2.items(),key=lambda x:x[1],reverse=True)[0:1])
        predictions.append(p1.keys())
    predictions=pd.DataFrame({'predictions':predictions})
    predictions['original']=data.tags
    return predictions
