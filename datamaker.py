import pickle
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from nltk import wordpunct_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# (ds_train, ds_validation, ds_test), ds_info = tfds.load('cnn_dailymail', shuffle_files = True, split = ['train', 'validation', 'test'], with_info = True)

# df_train = tfds.as_dataframe(ds_train.take(-1), ds_info)
# df_train.head()

# df_val = tfds.as_dataframe(ds_validation.take(-1), ds_info)
# df_val.head()

# def decode_text(x):
#     return x.decode('utf-8')

# df_train.loc[:, 'article'] = df_train.loc[:, 'article'].progress_apply(decode_text)
# df_train.loc[:, 'highlights'] = df_train.loc[:, 'highlights'].progress_apply(decode_text)
# df_val.loc[:, 'article'] = df_val.loc[:, 'article'].progress_apply(decode_text)
# df_val.loc[:, 'highlights'] = df_val.loc[:, 'highlights'].progress_apply(decode_text)

# def length(x):
#     return len(wordpunct_tokenize(x))

# df_train['Article_Length'] = df_train['article'].progress_apply(length)
# df_train['Summary_Length'] = df_train['highlights'].progress_apply(length)
# df_val['Article_Length'] = df_val['article'].progress_apply(length)
# df_val['Summary_Length'] = df_val['highlights'].progress_apply(length)

# df_train.drop(df_train[df_train['Article_Length'] > 548].index, inplace = True)
# df_val.drop(df_val[df_val['Article_Length'] > 548].index, inplace = True)

def preprocess_text(free_text):    
    ft = free_text.strip()
    tokens = wordpunct_tokenize(ft)
    tokens.append('_end_')
    tokens.insert(0, '_start_')
    return ' '.join(tokens)

tra = pd.read_csv('training_articles.csv')
trs = pd.read_csv('training_summaries.csv')
df_train = pd.concat([tra, trs], axis = 1)

tea = pd.read_csv('validation_articles.csv')

tes = pd.read_csv('validation_summaries.csv')
df_val = pd.concat([tea, tes], axis = 1)

df_train['preprocessed_articles'] = df_train['article'].progress_apply(preprocess_text)
df_train['preprocessed_summaries'] = df_train['highlights'].progress_apply(preprocess_text)
df_val['preprocessed_articles'] = df_val['article'].progress_apply(preprocess_text)
df_val['preprocessed_summaries'] = df_val['highlights'].progress_apply(preprocess_text)

print(df_train['preprocessed_articles'].iloc[:5])

# print(df_train.head())

tokenizer = Tokenizer(oov_token = '<oov>', filters='"#$%*+/<=>@[\\]^`{|}~\t\n')
tokenizer.fit_on_texts(df_train['preprocessed_articles'].tolist() + df_train['preprocessed_summaries'].tolist() + df_val['preprocessed_articles'].tolist() + df_val['preprocessed_summaries'].tolist())

print(tokenizer.word_index['_start_'])

print(tokenizer.word_index['_end_'])

with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

maxlen_article = 550
maxlen_summary = 50

train_article_sequences = tokenizer.texts_to_sequences(df_train['preprocessed_articles'].tolist())
train_summary_sequences = tokenizer.texts_to_sequences(df_train['preprocessed_summaries'].tolist())

validation_article_sequences = tokenizer.texts_to_sequences(df_val['preprocessed_articles'].tolist())
validation_summary_sequences = tokenizer.texts_to_sequences(df_val['preprocessed_summaries'].tolist())

X_train = pad_sequences(train_article_sequences, maxlen = maxlen_article, padding = 'post')
y_train = pad_sequences(train_summary_sequences, maxlen = maxlen_summary, padding = 'post')

X_val = pad_sequences(validation_article_sequences, maxlen = maxlen_article, padding = 'post')
y_val = pad_sequences(validation_summary_sequences, maxlen = maxlen_summary, padding = 'post')

with open('X_train.pickle', 'wb') as f:
    pickle.dump(X_train, f)

with open('X_val.pickle', 'wb') as f:
    pickle.dump(X_val, f)

with open('y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)

with open('y_val.pickle', 'wb') as f:
    pickle.dump(y_val, f)
