import pandas as pd
import time

from multilabelclassifier import multi_label_classification
from semanticsimilarity import semantic_similarity

df=pd.read_json('export-2018-02-09.json')
start=time.time()
print(semantic_similarity(df))
end=time.time()
print(end-start)
