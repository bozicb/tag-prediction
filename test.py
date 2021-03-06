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
        multi_label_classification(sample_df,False,test_size=1.0/len(sample_df))
        end=time.time()
        execution_times["method1_content"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        start=time.time()
        multi_label_classification(sample_df,True,test_size=1.0/len(sample_df))
        end=time.time()
        execution_times["method1_content+books"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        #Method 2
        start=time.time()
        word_frequencies(sample_df,one_run=True)
        end=time.time()
        execution_times["method2"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        #Method 3
        start=time.time()
        nearest_centroid(sample_df,False,test_size=1.0/len(sample_df))
        end=time.time()
        execution_times["method3_content"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        start=time.time()
        nearest_centroid(sample_df,True,test_size=1.0/len(sample_df))
        end=time.time()
        execution_times["method3_content+books"][i]=end-start
        execution_times.to_csv("performance_test.csv")
        #Method 4
        for k in range(5):
            start=time.time()
            knn(sample_df,k+1,False,one_run=True)
            end=time.time()
            execution_times["method4_content_k"+str(k+1)][i]=end-start
            execution_times.to_csv("performance_test.csv")
            start=time.time()
            knn(sample_df,k+1,True,one_run=True)
            end=time.time()
            execution_times["method4_content+books_k"+str(k+1)][i]=end-start
            execution_times.to_csv("performance_test.csv")

def recall(df,m1=None,m1b=None,m2=None,m3=None,m3b=None,m41=None,m4b1=None,m42=None,
           m4b2=None,m43=None,m4b3=None,m44=None,m4b4=None,m45=None,m4b5=None):
    recall=pd.Series(index=['method1_content','method1_content+books','method2',
                            'method3_content','method3_content+books','method4_content_k1',
                            'method4_content+books_k1','method4_content_k2',
                            'method4_content+books_k2','method4_content_k3',
                            'method4_content+books_k3','method4_content_k4',
                            'method4_content+books_k4','method4_content_k5',
                            'method4_content+books_k5'])
    if m1 is None:
        m1=multi_label_classification(df,False)
        m1.to_csv('recall/method1_content.csv')
        m1b=multi_label_classification(df,True)
        m1b.to_csv('recall/method1_content+books.csv')
        m2=word_frequencies(df)
        m2.to_csv('recall/method2.csv')
        m3=nearest_centroid(df,False)
        m3.to_csv('recall/method3_content.csv')
        m3b=nearest_centroid(df,True)
        m3b.to_csv('recall/method3_content+books.csv')
        m41=knn(df,1,False)
        m41.to_csv('recall/method4_content_k1.csv')
        m4b1=knn(df,1,True)
        m4b1.to_csv('recall/method4_content+books_k1.csv')
        m42=knn(df,2,False)
        m42.to_csv('recall/method4_content_k2.csv')
        m4b2=knn(df,2,True)
        m4b2.to_csv('recall/method4_content+books_k2.csv')
        m43=knn(df,3,False)
        m43.to_csv('recall/method4_content_k3.csv')
        m4b3=knn(df,3,True)
        m4b3.to_csv('recall/method4_content+books_k3.csv')
        m44=knn(df,4,False)
        m44.to_csv('recall/method4_content_k4.csv')
        m4b4=knn(df,4,True)
        m4b4.to_csv('recall/method4_content+books_k4.csv')
        m45=knn(df,5,False)
        m45.to_csv('recall/method4_content_k5.csv')
        m4b5=knn(df,5,True)
        m4b5.to_csv('recall/method4_content+books_k5.csv')
    for ix,m in enumerate([m1,m1b,m2,m3,m3b,m41,m4b1,m42,m4b2,m43,m4b3,m44,m4b4,m45,m4b5]):
        x=[]
        for i in range(len(m['predictions'])):
            x.append(len([e for e in m['original'].iloc[i] if e in m['predictions'].iloc[i]])/
                 float(len(m['original'].iloc[i])))
        recall[ix]=sum(x)/float(len(x))
        recall.to_csv('recall_test.csv')

df=pd.read_json('export-2018-02-09.json')
'''
m1=pd.read_csv('recall/method1_content.csv')
m1b=pd.read_csv('recall/method1_content+books.csv')
m2=pd.read_csv('recall/method2.csv')
m3=pd.read_csv('recall/method3_content.csv')
m3b=pd.read_csv('recall/method3_content+books.csv')
m41=pd.read_csv('recall/method4_content_k1.csv')
m4b1=pd.read_csv('recall/method4_content+books_k1.csv')
m42=pd.read_csv('recall/method4_content_k2.csv')
m4b2=pd.read_csv('recall/method4_content+books_k2.csv')
m43=pd.read_csv('recall/method4_content_k3.csv')
m4b3=pd.read_csv('recall/method4_content+books_k3.csv')
m44=pd.read_csv('recall/method4_content_k4.csv')
m4b4=pd.read_csv('recall/method4_content+books_k4.csv')
m45=pd.read_csv('recall/method4_content_k5.csv')
m4b5=pd.read_csv('recall/method4_content+books_k5.csv')
'''
performance(df)
#recall(df)
