from tagging.helper import tag_word_frequencies
from tagging.helper import clean
from tagging.helper import sentence_similarity

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
        elif len(similarity)>0:
            p=dict((x,y) for x,y in sorted(similarity.items(),
                            key=lambda x:x[1],reverse=True)[0:len(dg)])
        else:
            similarity={key:value for key,value in similarity.items()}
            p=dict((x,y) for x,y in sorted(similarity.items(),
                            key=lambda x:x[1],reverse=True)[0:1])
        predictions.append(p)
    predictions=pd.DataFrame({'predictions':predictions})
    predictions['original']=data.tags
    return predictions

