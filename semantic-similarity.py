import pandas as pd
import re
import string

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag,word_tokenize

#English Stop Words
stop=set(stopwords.words('english'))
#Punctuation 
exclude=set(string.punctuation)
#Replace functional approach
repls=("(",""),(")",""),("'s","")
#Exclusion Words
words_ex=[]
#Lemmatization
lemma=WordNetLemmatizer()

def penn_to_wn(tag):
    #Convert from a Penn Treebank tag to a simplified Wordnet tag
    if tag.startswith('N'):
        return 'n'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('J'):
        return 'a'
    elif tag.startswith('R'):
        return 'r'
    else:
        return None
 
def tagged_to_synset(word,tag):
    wn_tag=penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word,wn_tag)[0]
    except:
        return None

def clean(text):
    #Remove specific chars
    rem1=re.sub("[!#*%:1234567890?,.;&-]","",text)
    #Remove non ASCII chars
    rem2=re.sub(r'[^\x00-\x7f]',r' ',rem1)
    #Replace using functional approach
    rem3=reduce(lambda a,kv:a.replace(*kv),repls,rem2)   
    #Exclude stop words
    stop_free=" ".join([i for i in rem2.lower().split() if i not in
                        stop])
    #Exclude Punctuation
    punc_free=''.join(i for i in stop_free if i not in exclude)
    #Exclude words
    cust_words=" ".join(i for i in punc_free.split() if i not in
                        words_ex)
    #Exclude numbers
    num_free=''.join([i for i in cust_words if not i.isdigit()])
    #Lemmatization
    normalized=" ".join(lemma.lemmatize(word) for word in
                          num_free.split())
    return normalized
        
def top_words_tags(df):
    list_of_tags=df.tags.tolist()
    #keys are unique tags
    keys=list(set([item for sublist in list_of_tags for item in
                   sublist]))
    series_of_tags = pd.Series([str(list_of_tags[x]).strip('[]') for x in
                                range(len(list_of_tags))])
    word_freq={}
    for key in keys:
        #for every key: 
        #(1) get a list of all tags where the key is contained, 
        #(2) get value counts of tags with the key / all value counts
        #    for that key
        #(3) add frequencies to dictionary
        list_of_words_for_tag=[item for sublist in 
                          [word_tokenize(doc) for doc in df[
                              series_of_tags.str.contains(
                                  key,case=False)].iloc[:,2].tolist()] 
                          for item in sublist]
        all_word_frequencies=((pd.Series(list_of_words_for_tag).
                               value_counts())/
                              len(list_of_words_for_tag)).to_dict()
        word_freq[key]=all_word_frequencies
    return word_freq

#This algorithm is proposed by Mihalcea et al. in the paper "Corpus-based and Knowledge-based Measures
#of Text Semantic Similarity" (https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf)

def sentence_similarity(text,words):
    # Get the synsets for the tagged words
    text_synsets=[tagged_to_synset(*tagged_text) for tagged_text in
              text]
    word_synsets=[tagged_to_synset(*tagged_word) for tagged_word in
                  words]
 
    # Filter out the Nones
    text_synsets = [text for text in text_synsets if text]
    word_synsets = [word for word in word_synsets if word]
 
    score, count = 0.0, 0

    #For each word in the first sentence
    for text in text_synsets:
        # Get the similarity value of the most similar word in the
        # other sentence
        best_score = max([text.path_similarity(word) for word in
                          word_synsets])
        #Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    #Average the values
    if count!=0:
        score /= count
    else:
        score=0
    return score

def pred3(text,data):
    meta={}
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

def method3(data):
    predictions={}
    word_freq=top_words_tags(data)
    tags=word_freq.keys()
    texts=[]
    words=[]
    similarity={}
    for text in list(data['content']):
        texts.append(pos_tag(word_tokenize(clean(text))))
    for tag in tags:
        words.append(pos_tag(word_tokenize(
            " ".join(word_freq[tag].keys()))))
    for text in texts:
        for word in words:
            similarity[tag]=sentence_similarity(text,word)
        #dg={key:value for key,value in similarity.items() if value>=0.2}
    print(similarity)
    #return predictions

df=pd.read_json('export-2018-02-09.json')
method3(df)
