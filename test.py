import pandas as pd
import time

from multilabel import multi_label_classification
from semanticsimilarity import semantic_similarity
from wordfrequencies import word_frequencies
from nearestcentroid import nearest_centroid
from knn import knn

def performance(df):
    execution_times=pd.DataFrame(index=[10,20,30,40,50,60,70,80,90,100],
        columns=['method1_content','method1_content+books','method2','method3_content',
                 'method3_content+books','method4_content_k1','method4_content+books_k1',
                 'method4_content_k2','method4_content+books_k2','method4_content_k3',
                 'method4_content+books_k3','method4_content_k4','method4_content+books_k4',
                 'method4_content_k5','method4_content+books_k5'])
    for i in xrange(10,101,10):
        #Method 1
        sample_df = df.sample(frac=i/100.0)
        start=time.time()
        multi_label_classification(sample_df,False)
        end=time.time()
        execution_times["method1_content"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        start=time.time()
        multi_label_classification(sample_df,True)
        end=time.time()
        execution_times["method1_content+books"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        #Method 2
        start=time.time()
        word_frequencies(sample_df)
        end=time.time()
        execution_times["method2"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        #Method 3
        start=time.time()
        nearest_centroid(sample_df,False)
        end=time.time()
        execution_times["method3_content"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        start=time.time()
        nearest_centroid(sample_df,True)
        end=time.time()
        execution_times["method3_content+books"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        #Method 4
        for k in range(5):
            start=time.time()
            knn(sample_df,k+1,False)
            end=time.time()
            execution_times["method4_content_k"+str(k+1)][i]=end-start
            execution_times.to_csv("performance_test.csv")
            start=time.time()
            knn(sample_df,k+1,True)
            end=time.time()
            execution_times["method4_content+books_k"+str(k+1)][i]=end-start
            execution_times.to_csv("performance_test.csv")
    display(execution_times)

def recall(df):
    recall=pd.Series(index=['method1_content','method1_content+books','method2',
                            'method3_content','method3_content+books','method4_content_k1',
                            'method4_content+books_k1','method4_content_k2',
                            'method4_content+books_k2','method4_content_k3',
                            'method4_content+books_k3','method4_content_k4',
                            'method4_content+books_k4','method4_content_k5',
                            'method4_content+books_k5'])
    m1=multi_label_classification(df,False)
    m1b=multi_label_classification(df,True)
    m2=word_frequencies(df)
    m3=nearest_centroid(df,False)
    m3b=nearest_centroid(df,True)
    m41=knn(df,1,False)
    m4b1=knn(df,1,True)
    m42=knn(df,2,False)
    m4b2=knn(df,2,True)
    m43=knn(df,3,False)
    m4b3=knn(df,3,True)
    m44=knn(df,4,False)
    m4b4=knn(df,4,True)
    m45=knn(df,5,False)
    m4b5=knn(df,5,True)
    for ix,m in enumerate([m1,m1b,m2,m3,m3b,m41,m4b1,m42,m4b2,m43,m4b3,m44,m4b4,m45,m4b5]):
        x=[]
        for i in range(len(result['predictions'])):
            x.append(len([e for e in result['predictions'][i] if e in result['original'][i]])/
                 float(len(result['original'][i])))
        recall[ix]=sum(x)/float(len(x))
        recall.to_csv('recall_test.csv')

df=pd.read_json('export-2018-02-09.json')
performance(df)
recall(df)

