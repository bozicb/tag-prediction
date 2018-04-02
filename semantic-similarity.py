import pandas as pd
import string
import re
import time

from sklearn.model_selection import train_test_split
from nltk import pos_tag,word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

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
	rem1 = re.sub("[!#*%:1234567890?,.;&-]", "", text)
    #Remove non ASCII chars
	rem2 = re.sub(r'[^\x00-\x7f]', r' ', rem1)
    #Replace using functional approach
	rem3 = reduce(lambda a, kv: a.replace(*kv), repls, rem2)   
    #Exclude stop words
	stop_free = " ".join([i for i in rem2.lower().split() if i not in stop])
    #Exclude Punctuation
	punc_free = ''.join(i for i in stop_free if i not in exclude)
    #Exclude words
	cust_words = " ".join(i for i in punc_free.split() if i not in words_ex)
    #Exclude numbers
	num_free = ''.join([i for i in cust_words if not i.isdigit()])
    #Lemmatization
	normalized = " ".join(lemma.lemmatize(word) for word in num_free.split())
	return normalized

def top_words_tags(df,numw,pct):
    #df: dataframe original format
    #numw: number of words according to the frequency 
    #pct: max percentage
    ltag=df.tags.tolist()
    ltag1 = [item for sublist in ltag for item in sublist]
    s = pd.Series([ str(ltag[x]).strip('[]') for x in range(len(ltag))])
    dict_freq={}
    keys=list(set(ltag1))
    for i in keys:
        tg1=df[s.str.contains(i,case=False)]
        text=tg1.iloc[:,2]
        text_list=text.tolist()
        doc_clean = [clean(doc).split() for doc in text_list]
        l1 = [item for sublist in doc_clean for item in sublist]
        temp1=((pd.Series(l1).value_counts()[0:numw])/sum(pd.Series(l1).value_counts()[0:numw].to_dict().values())).to_dict()
        #temp2=sorted(temp1.items(), key=lambda x:x[1],reverse=True)[0:ind(sorted(temp1.values(),reverse=True),pct)]
        #dict_freq[i]=dict((x, y) for x, y in temp2)
        dict_freq[i]=temp1
    return dict_freq

def penn_to_wn(tag):
    #""" Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

#This algorithm is proposed by Mihalcea et al. in the paper "Corpus-based and Knowledge-based Measures
#of Text Semantic Similarity" (https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf)

def sentence_similarity(sentence1, sentence2):
    # compute the sentence similarity using Wordnet
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    if count!=0:
        score /= count
    else:
        score=0
    return score
    
def pred3(text,df):
    meta={}
    temp1=top_words_tags(df,20,80)
    tags=temp1.keys()
    for i in tags:
        str1=" ".join(temp1[i].keys())
        meta[i]=sentence_similarity(clean(text),str1)
    dg={ key:value for key, value in meta.items() if value >= 0.2}
    if len(dg)!=0:
        if len(dg)>=3:
            p1=dict((x, y) for x, y in sorted(dg.items(), key=lambda x:x[1],reverse=True)[0:3])
        else:
            p1=dict((x, y) for x, y in sorted(dg.items(), key=lambda x:x[1],reverse=True)[0:len(dg)])
    else:
        dg2 = { key:value for key, value in meta.items() if value < 0.2}
        p1=dict((x, y) for x, y in sorted(dg2.items(), key=lambda x:x[1],reverse=True)[0:1])
    return p1.keys()
    
def semantic_similarity(data):
    train,test=train_test_split(data,test_size=.2)
    train=train.reset_index().drop('index', axis=1)
    test=test.reset_index()
    lx=list(test['content'])
    nl=len(lx)
    lp=[]
    for i in lx:
        lp.append(pred3(i,train))
    return lp

df=pd.read_json('export-2018-02-09.json')
start=time.time()
print(pd.Series(semantic_similarity(df)))
end=time.time()
print("time: "+str(end-start))
