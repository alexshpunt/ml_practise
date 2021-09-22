import numpy as np 
import pandas as pd

from bs4 import BeautifulSoup
def clean_tags(data): return data.apply(BeautifulSoup).apply(lambda s: s.get_text())

def get_sentences(data, tokenizer): return data.apply(lambda r: tokenizer.tokenize(r.strip()))

import re 
def sentence_keep_letters_and_digits(data): 
    return data.apply(lambda ss: [re.sub("[^a-zA-Z0-9]", " ", s).strip() for s in ss])
def keep_letters_and_digits(data): 
    return data.apply(lambda s: re.sub("[^a-zA-Z0-9]", " ", s).strip())

def sentences_to_words(data): return data.apply(lambda ss: [s.lower().split() for s in ss])
def to_words(data): return data.apply(lambda s: s.lower().split())

from nltk.corpus import stopwords
def sentences_remove_stop_words(data):
    stops = set(stopwords.words('english'))
    return [[w for w in s if w not in stops] for s in data ]

def remove_stop_words(data):
    print(data)
    stops = set(stopwords.words('english'))
    return [w for w in data if w not in stops]

def merge_into_sentences(data):
    return [' '.join([w for w in s]) for s in data ]

import nltk
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def process_data(data, use_sentences=False, remove_stopwords=True): 
    data = clean_tags(data)
    if use_sentences: 
        data = get_sentences(data, tokenizer)
        data = sentence_keep_letters_and_digits(data)
        data = sentences_to_words(data)
    else: 
        data = keep_letters_and_digits(data)
        data = to_words(data)
    if remove_stopwords:
        data = sentences_remove_stop_words(data)
    return list(data)

def make_features(data, model, use_sentences):
    if use_sentences: return make_features_sentences(data, model)

    shape = (len(data), model.vector_size)
    print(shape)
    f = np.zeros(shape)
    for i, review in enumerate(data): 
        for w in review: 
            try: 
                vec = model[w]
            except KeyError:
                continue
            f[i,:] += vec
        if len(review) > 0: f[i,:] /= len(review)
    return f

def make_features_sentences(data, model):
    shape = (len(data), model.vector_size)
    print(shape)
    f = np.zeros(shape)
    for i, review in enumerate(data): 
        w_count = 0
        for s in review: 
            w_count =+ len(s)
            for w in s: 
                try: 
                    vec = model[w]
                except KeyError:
                    continue
                f[i,:] += vec
        if w_count > 0: f[i,:] /= w_count
    return f

from matplotlib.pyplot import * 
def measure_model(model, test_data, target_test_data, color='g'):
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
    pred = model.predict_proba(test_data)[:,1]
    acc = accuracy_score(target_test_data, pred > 0.5)
    auc = roc_auc_score(target_test_data, pred)
    fpr, tpr, thr = roc_curve(target_test_data, pred)
    plot(fpr, tpr, color)
    xlabel('false positive')
    ylabel('true positive')
    return {
        'acc':acc,
        'auc':auc,
    }

from os.path import join
data_path = 'data/nlp/raw/kaggle/word2vec-nlp-tutorial/'
raw_train_data = pd.read_csv(join(data_path, 'labeledTrainData.tsv'), delimiter='\t', quoting=3)
submission_data = pd.read_csv(join(data_path, 'testdata.tsv'), delimiter='\t', quoting=3)
raw_unlabeled_train_data = pd.read_csv(join(data_path, 'unlabeledTrainData.tsv'), delimiter='\t', quoting=3)

from sklearn.model_selection import train_test_split
split_data = train_test_split(raw_train_data.review, raw_train_data.sentiment)
train_data, test_data, target_train_data, target_test_data = split_data

use_sentences = False
processed_train_data = process_data(train_data, use_sentences)
processed_raw_train_data = process_data(raw_train_data.review, use_sentences)
processed_test_data = process_data(test_data, use_sentences)
processed_sumbission_data = process_data(submission_data.review)
processed_unlabeled_train_data = process_data(raw_unlabeled_train_data.review, use_sentences)

train_tokens = processed_unlabeled_train_data
train_tokens.extend(processed_raw_train_data)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

use_w2v = True
if use_w2v: 
    use_built_in_model = False
    if use_built_in_model:
        import gensim.downloader as api
        word_vectors = api.load("glove-wiki-gigaword-100")
    else: 
        from gensim.models.word2vec import Word2Vec
        load_w2v_model = False
        model_path = join(data_path, 'w2v.model')
        if not load_w2v_model: 
            model = Word2Vec(train_tokens, vector_size=300, window=5, workers=16, epochs=25, min_count=30)
            model.save(model_path)
        else: 
            model = Word2Vec.load(model_path)
        word_vectors = model.wv 

    train_features = make_features(processed_train_data, word_vectors, use_sentences)
    test_features = make_features(processed_test_data, word_vectors, use_sentences)
    submission_features = make_features(processed_sumbission_data, word_vectors, use_sentences)
else: 
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer()
    v.fit(train_tokens)
    processed_train_data = merge_into_sentences(processed_train_data)
    processed_test_data = merge_into_sentences(processed_test_data)
    train_features = v.transform(processed_train_data)
    test_features = v.transform(processed_test_data)
    submission_features = v.transform(processed_sumbission_data)

from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=300, verbose=True, n_jobs=-1)
m.fit(train_features, target_train_data)
print(measure_model(m, test_features, target_test_data))

submission_pred = m.predict(submission_features)
submission_data['id'] = submission_data['id'].apply(lambda x: x.replace('\"', ''))
submission_data['sentiment'] = submission_pred
submission_data.drop('review', axis=1).to_csv(join(data_path, 'sampleSubmission.csv'), index=False)