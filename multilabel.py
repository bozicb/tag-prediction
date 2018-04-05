import pandas as pd
import numpby as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from scipy.sparse import coo_matrix,hstack

def multi_label_classification(data,books=True,test_size=.2):
    train,test=train_test_split(data,test_size=test_size)
    train_x_content=train['content'].as_matrix()
    test_x_content=test['content'].as_matrix()
    train_y=train['tags'].as_matrix()
    test_y=test['tags'].as_matrix()
    
    # Transform training tags into a multi-label binary format
    mlb=MultiLabelBinarizer(classes=list(set([item for sublist in df.tags.tolist()
                                                for item in sublist])))
    train_labels=mlb.fit_transform(train_y)
    # Transform test-tags into a multi-label binary format
    test_labels=mlb.fit_transform(test_y)
    
    # Applying CountVectorizer on character n-grams
    count_vectorizer_content=CountVectorizer(stop_words="english",tokenizer=tokenize,
                                             ngram_range=(1, 3),max_features=10000,
                                             analyzer="word")

    # Learn and transform train-description
    cv_train_x_content=count_vectorizer_content.fit_transform(train_x_content)
    cv_test_x_content=count_vectorizer_content.transform(test_x_content)
    
    # Applying TfIdfVectorizer on individual words(specifically for tri-grams range)
    tfidf_vectorizer_content=TfidfVectorizer(stop_words="english", tokenizer=tokenize,
                                             ngram_range=(1, 3),max_features=10000,
                                             analyzer="word")
    
    # Learn and transform train-description
    tv_train_x_content=tfidf_vectorizer_content.fit_transform(train_x_content)
    tv_test_x_content=tfidf_vectorizer_content.transform(test_x_content)
    
    # GradientBoostingClassifier with parameter tuning
    params={"n_estimators":170,"max_depth":5,"random_state":10,"min_samples_split":4,
            "min_samples_leaf":2}
    classifier=OneVsRestClassifier(GradientBoostingClassifier(**params))
    
    cv_train_x=cv_test_x=tv_train_x=tv_test_x=None
    
    # Add Books
    if books:
        train_x_books=train['book_name'].apply(lambda x:x[0]).as_matrix()
        test_x_books=test['book_name'].apply(lambda x:x[0]).as_matrix()
        count_vectorizer_books=CountVectorizer(stop_words="english",tokenizer=tokenize,
                                               ngram_range=(1,3),max_features=10000,
                                               analyzer="word")
        cv_train_x_books=count_vectorizer_books.fit_transform(train_x_books)
        cv_test_x_books=count_vectorizer_books.transform(test_x_books)
        tfidf_vectorizer_books=TfidfVectorizer(stop_words="english",tokenizer=tokenize,
                                               ngram_range=(1,3),max_features=10000,
                                               analyzer="word")
        tv_train_x_books=tfidf_vectorizer_books.fit_transform(train_x_books)
        tv_test_x_books=tfidf_vectorizer_books.transform(test_x_books)
        cv_train_x=hstack([cv_train_x_books,cv_train_x_content])
        cv_test_x=hstack([cv_test_x_books,cv_test_x_content])
        tv_test_x=hstack([tv_test_x_books,tv_test_x_content])
        tv_train_x=hstack([tv_train_x_books,tv_train_x_content])
    else:
        cv_train_x,cv_test_x,tv_train_x,tv_test_x=cv_train_x_content,cv_test_x_content,
        tv_train_x_content,tv_test_x_content
    
    # Generate predictions using counts
    classifier.fit(cv_train_x,train_labels)
    filename1='count_vectorizer_model.pkl'
    pickle.dump(classifier,open(filename1,'wb'))
    
    # Generate predictions using tf-idf representation
    classifier.fit(tv_train_x,train_labels)
    filename2='tfidf_vectorizer_model.pkl'
    pickle.dump(classifier,open(filename2,'wb'))
    
    # Import trained models
    count_vectorizer_model,tfidf_vectorizer_model=joblib.load(filename1),
    joblib.load(filename2)
    
    # Prediction assignment
    cv_pred,tv_pred=count_vectorizer_model.predict(cv_test_x.toarray()),
    tfidf_vectorizer_model.predict(tv_test_x.toarray())
    
    # Combine predictions and map labels
    combined_pred=np.where((cv_pred+tv_pred)!=0,mlb.classes_,"")
    
    # Load the array into a DataFrame constructor and join non-empty strings
    predictions=pd.DataFrame(combined_pred).apply(join_strings,axis=1).to_frame("predicted")
    predictions['predicted']=predictions['predicted'].apply(lambda x:x.split())
    predictions=pd.concat([test['tags'].reset_index(),predictions['predicted']],axis=1,
                          keys=['original','predicted'],ignore_index=True)
    predictions.drop(0,axis=1,inplace=True)
    predictions.columns=['original','predicted']
    return predictions
