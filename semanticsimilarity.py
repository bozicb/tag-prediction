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
    rem1=re.sub("[!#*%:1234567890?,.;&-]","",text)
    #Remove non ASCII chars
    rem2=re.sub(r'[^\x00-\x7f]',r' ',rem1)
    #Replace using functional approach
    rem3=reduce(lambda a,kv:a.replace(*kv),repls,rem2)   
    #Exclude stop words
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
    normalized=" ".join(lemma.lemmatize(word) for word
                        in num_free.split())
    return normalized

def tag_word_frequencies(df):
    list_of_tags=df.tags.tolist()
    #keys are unique tags
    keys=list(set([item for sublist in list_of_tags
                   for item in sublist]))
    series_of_tags=pd.Series([str(list_of_tags[x]).strip('[]')
                                for x in range(len(list_of_tags))])
    word_freq={}
    for key in keys:
        #for every key: 
        #(1) get a list of all tags where the key is contained, 
        #(2) get value counts of tags with the key / all value counts
        #    for that key
        #(3) add frequencies to dictionary
        list_of_words_for_tag=[item for sublist in 
            [word_tokenize(doc) for doc in
             df[series_of_tags.str.contains(key,case=False)].iloc[:,2]
             .tolist()] 
            for item in sublist]
        all_word_frequencies=((pd.Series(list_of_words_for_tag)
                               .value_counts())/
                              len(list_of_words_for_tag)).to_dict()
        word_freq[key]=all_word_frequencies
    return word_freq

def tagged_to_synset(word, tag):
    if tag.startswith('N'):wn_tag='n' 
    elif tag.startswith('V'):wn_tag='v'
    elif tag.startswith('J'):wn_tag='a'
    elif tag.startswith('R'):wn_tag='r'
    else:wn_tag=None
    if wn_tag is None:return None
    try:return wn.synsets(word, wn_tag)[0]
    except:return None

#This algorithm is proposed by Mihalcea et al. in the paper
#"Corpus-based and Knowledge-based Measures of Text Semantic Similarity"
#(https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf)

def sentence_similarity(sentence1, sentence2):
    #Compute the sentence similarity using Wordnet
    #Tokenize and tag
    sentence1=pos_tag(word_tokenize(sentence1))
    sentence2=pos_tag(word_tokenize(sentence2))
 
    #Get the synsets for the tagged words
    synsets1=[tagged_to_synset(*tagged_word) for tagged_word
              in sentence1]
    synsets2=[tagged_to_synset(*tagged_word) for tagged_word
              in sentence2]
 
    #Filter out the Nones
    synsets1=[ss for ss in synsets1 if ss]
    synsets2=[ss for ss in synsets2 if ss]
 
    score,count=0.0,0
 
    #For each word in the first sentence
    for synset in synsets1:
        #Get the similarity score
        best_score=max([synset.path_similarity(ss) for ss in synsets2])
        #Check that the similarity could have been computed
        if best_score is not None:
            score+=best_score
            count+=1
    #Average the values
    if count!=0:score/=count
    else:score=0
    return score
    
def semantic_similarity(data):
    data=data.reset_index().drop('index', axis=1)
    texts=list(data['content'])
    word_freqs=tag_word_frequencies(data)
    tags=word_freqs.keys()
    predictions=[]
    for text in texts:
        text=clean(text)
        similarity={}
        for tag in tags:
            similarity[tag]=sentence_similarity(text," ".join(
                word_freqs[tag].keys()))
        similarity={key:value for key,value in similarity.items()
                    if value >=.2}
        if len(similarity)>=3:
            p=dict((x,y) for x,y in sorted(similarity.items(),
                            key=lambda x:x[1],reverse=True)[0:3])
        elif len(dg)>0:
            p=dict((x,y) for x,y in sorted(similarity.items(),
                            key=lambda x:x[1],reverse=True)[0:len(dg)])
        else:
            similarity={key:value for key,value in similarity.items()
                        if value < 0.2}
            p=dict((x,y) for x,y in sorted(similarity.items(),
                            key=lambda x:x[1],reverse=True)[0:1])
        predictions.append(p1)
    return predictions

train,test=train_test_split(df,test_size=test_size)
    train=train.reset_index().drop('index',axis=1)
    texts=list(data['content'])
    tags=list(set([item for sublist in train.tags.tolist() for item in sublist]))
    s=pd.Series([str(train.tags.tolist()[x]).strip('[]')
                 for x in range(len(train.tags.tolist()))])
    predictions=[]
    if include_books:
        books=list(data['book_name'].apply(lambda x:x[0]))
        train_text=train['content'].as_matrix()
        train_book=train['book_name'].apply(lambda x:x[0])
        vec_t=TfidfVectorizer(stop_words="english",tokenizer=word_tokenize,ngram_range=(1,3),
                              max_features=10000,analyzer="word")
        vec_b=TfidfVectorizer(stop_words="english",tokenizer=word_tokenize,ngram_range=(1,3),
                              max_features=10000,analyzer="word")
        vec_text_x=vec_t.fit_transform(train_text)
        vec_book_x=vec_b.fit_transform(train_book)
        vec_all=hstack([vec_book_x,vec_text_x]).toarray()
        for text,book in zip(texts,books):
            vec_text=vec_t.transform([clean(text)])
            vec_book=vec_b.transform([clean(book)])
            vec=hstack([vec_book,vec_text]).toarray()
            dict_sim={}
            for tag in tags:
                tg1=train[s.str.contains(tag,case=False)]
                ind1=tg1.index.values
                X=vec_all[ind1]
                kmeans_model=KMeans(n_clusters=1).fit(X)
                dict_sim[tag]=cosine_similarity(vec,np.array(
                    kmeans_model.cluster_centers_))[0,0]
            dg={key:value for key,value in dict_sim.items() if value>=0.1}
            if len(dg)>=3:p1=dict((x,y) for x,y
                        in sorted(dg.items(),key=lambda x:x[1],reverse=True)[0:3])
            elif len(dg)>0:p1=dict((x,y) for x,y
                        in sorted(dg.items(),key=lambda x:x[1],reverse=True)[0:len(dg)])
            else:p1=dict((x,y) for x,y
                         in sorted(dict_sim.items(), key=lambda x:x[1],reverse=True)[0:1])
            predictions.append(p1.keys())
    else:
        train_texts=train['content'].tolist()
        vectorizer=CountVectorizer(stop_words='english',tokenizer=word_tokenize,
                                   ngram_range=(1,1),max_features=10000,analyzer='word')
        train_vectorizer=vectorizer.fit_transform(train_texts)
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
            if len(dg)>=3:p1=dict((x,y) for x,y
                        in sorted(dg.items(),key=lambda x:x[1],reverse=True)[0:3])
            elif len(dg)>0:p1=dict((x,y) for x,y
                        in sorted(dg.items(),key=lambda x:x[1],reverse=True)[0:len(dg)])
            else:p1=dict((x,y) for x,y
                         in sorted(dict_sim.items(), key=lambda x:x[1],reverse=True)[0:1])
            predictions.append(p1.keys())    
    predictions=pd.DataFrame({'predictions':predictions})
    predictions['original']=data.tags
    return predictionsdf
