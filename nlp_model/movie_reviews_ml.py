from os import replace
import re
from numpy import product
from numpy.linalg import norm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from shared.data_utility import * 
from matplotlib.pyplot import *
from itertools import product

data = pd.read_csv('data/study/movie_reviews.tsv', sep='\t')
data = data.drop('id', axis=1)

train_data, test_data = build_train_test_data(data, factor=0.8)
target_train_data = train_data.sentiment
target_test_data = test_data.sentiment
train_data = train_data.drop('sentiment', axis=1)
test_data = test_data.drop('sentiment', axis=1)

def build_nb_model(max_features=None, min_df=1, nb_alpha=1.0, vectorizer_type='Counter', return_pred=True):
    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
    if vectorizer_type == 'Tfidf': vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    features = vectorizer.fit_transform(train_data.review)
    test_features = vectorizer.transform(test_data.review)

    model = MultinomialNB(alpha=nb_alpha)
    model.fit(features, target_train_data)
    pred = model.predict_proba(test_features)
    return model, vectorizer, {
        'max_features': max_features,
        'min_df': min_df,
        'nb_alpha': nb_alpha,
        'auc': roc_auc_score(target_test_data, pred[:,1]),
        'pred': pred if return_pred else None,
    }

def build_rf_model(max_features=None, min_df=1):
    vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', 
        min_df=min_df, max_features=max_features, norm='l2')
    features = vectorizer.fit_transform(train_data.review)
    test_features = vectorizer.transform(test_data.review)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=True)
    model.fit(features, target_train_data)
    pred = model.predict_proba(test_features)
    params = {
        'max_features': max_features,
        'min_df': min_df
    }
    return model, vectorizer, pred, params

param_values = {
    'max_features': [None],
    'min_df': [1,2,3],
    'nb_alpha': [0.01, 0.1, 1.0],
    'vectorizer_type': ['Counter', 'Tfidf']
}

#The best setup found by bruteforce 
#29	NaN	1	1.0	0.933601
def bruteforce_hyperparams():
    results = []
    for p in product(*param_values.values()):
        params = zip(param_values.keys(), p)
        params = dict(params)
        res = build_nb_model(**params)
        results.append(res)
        print(res)
    return pd.DataFrame(results)

def perf(y, pred, color='g', ann=True):
    pred = pred[:,1]
    acc = accuracy_score(y, pred > 0.5)
    auc = roc_auc_score(y, pred)
    print(f'acc: {acc} auc: {auc}')
    fpr, tpr, thr = roc_curve(y, pred)
    plot(fpr, tpr, color, linewidth='3')
    xlabel('false positive')
    ylabel('true positive')
    if ann: 
        annotate("Acc: %0.2f" % acc, (0.1,0.8), size=14)
        annotate("AUC: %0.2f" % auc, (0.1,0.7), size=14)

# model1, vectorizer1, params1 = build_nb_model()
# pred1 = params1['pred']
# perf(target_test_data, pred1, ann=False)

# model2, vectorizer2, params2 = build_nb_model(vectorizer_type='Tfidf')
# pred2 = params2['pred']
# perf(target_test_data, pred2, color='r', ann=False)

# model3, vectorizer3, pred3, params3 = build_rf_model()
# perf(target_test_data, pred3, ann=False)

from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, preprocess_documents

def featurize_w2v(vectorizer, reviews):
    f = np.zeros((len(reviews), vectorizer.vector_size))
    for i,s in enumerate(reviews): 
        for w in s: 
            try: 
                vec = vectorizer.wv[w]
            except KeyError:
                continue
            f[i,:] += vec
        f[i,:] /= len(s)
    return f

train_reviews = list(train_data.review)#.apply(remove_stopwords)
train_tokens = preprocess_documents(train_reviews)
vectorizer4 = Word2Vec(sentences=train_tokens, vector_size=300, window=5, min_count=3, workers=16)
features4 = featurize_w2v(vectorizer4, train_tokens)

model4 = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=True)
model4.fit(features4, target_train_data)

test_reviews = list(test_data.review)#.apply(remove_stopwords)
test_tokens = preprocess_documents(test_reviews)
test_features = featurize_w2v(vectorizer4, test_tokens)
pred4 = model4.predict_proba(test_features)

perf(target_test_data, pred4)

# train_reviews = [
#     'It is as good as its bad',
#     'You wont believe how annoying this film is', 
#     'It was so annoying I didnt watched it before',
#     'its not as bad as you might think'
# ]

# for review in reviews:
#     print(review, model.predict(vectorizer.transform([review]))[0])~