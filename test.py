import pandas as pd
import time

from multilabelclassifier import multi_label_classification
from semanticsimilarity import semantic_similarity
from wordfrequencies import word_frequencies
from nearestcentroid import nearest_centroid
from knn import knn

def performance_analysis(df):
    execution_times=pd.DataFrame(index=[10,20,30,40,50,60,70,80,90,100],
        columns=['method1_content','method1_content+books','method2','method3_content',
                 'method3_content+books','method4_content','method4_content+books'])
    for i in xrange(10,101,10):
        # Method 1
        sample_df = df.sample(frac=i/100)
        start1=time.time()
        method1(sample_df,False)
        end1=start2=time.time()
        method1(sample_df,True)
        end2=time.time()
        execution_times["method1_content"][i]=end1-start1
        execution_times.to_csv("performance_test.csv")
        execution_times["method1_content+books"][i]=end2-start2
        execution_times.to_csv("performance_test.csv")
        # Method 2
        start=time.time()
        method2(sample_df)
        end=time.time()
        execution_times["method2"][i]=end-start
        execution_times.to_csv("performance_test.csv")
    display(execution_times)

df=pd.read_json('export-2018-02-09.json')
start=time.time()
print(knn(df,2))
end=time.time()
print(end-start)
