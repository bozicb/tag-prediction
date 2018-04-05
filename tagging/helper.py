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
