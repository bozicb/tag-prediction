def knn(data,k):
    texts=list(df['content'])
    predictions=[]
    for text in texts:
        dist_cos={}
        lcs=[]
        df_x_cn = df['content'].as_matrix()
        tfidf_vectorizer_cn = TfidfVectorizer(stop_words="english", tokenizer=tokenize, ngram_range=(1, 3),max_features=10000, analyzer="word")
        tfidf_df_x_cn = tfidf_vectorizer_cn.fit_transform(df_x_cn)
        ltag=df.tags.tolist()
        vec_cn=tfidf_vectorizer_cn.transform([clean(text)])
        vec=vec_cn
        for i in range(df.shape[0]):
            dist_cos[i]={'tags':ltag[i],'cs':cosine_similarity(vec,tfidf_df_x_cn[i:i+1])[0,0]}
            lcs.append(cosine_similarity(vec,tfidf_df_x_cn[i:i+1])[0,0])
        r=np.array(lcs)
        rt=[ltag[j] for j in (r.argsort()[-k:][::-1]).tolist()]
        if k>1:
            a1=set(rt[1])
            for i in range(1,k):
                a2=a1&set(rt[i])
                a1=a2
            res=set(rt[0]) | a1
        else:
            res=set(rt[0])
        predictions.append(res)
    return predictions

start=time.time()
print(knn(df,2))
end=time.time()
display(end-start)
