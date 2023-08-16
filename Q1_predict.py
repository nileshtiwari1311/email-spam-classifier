import pandas as pd
import numpy as np
import math
import re
import string
import random
from random import sample
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import Q1_functions as Q1


# read the dataset
df = pd.read_csv('spam_ham_dataset.csv')
df = df[['text','label','label_num']]
print("Fitting Model....")

random.seed(40)
# divide dataset into train and test
train_df, test_df = Q1.train_test_split(df)
train_df['clean_text'] = train_df['text'].apply(Q1.text_preprocess)

# find the number of occurence of each word in the dataset
count_words = dict()
for text in train_df['clean_text']:
    words = text.split(" ")
    for word in words:
        count_words[word] = count_words.get(word, 0) + 1

# only choose top 1000 frequent words
max_words = 1000

top_k_words = sorted(count_words, reverse=True, key=count_words.get)[:max_words] # extract top k words with highest frequency
word_idx_map = dict(zip(top_k_words, [i for i in range(max_words)])) # map each word to a column index in data matrix
# data matrix where each row corresponds to a datapoint and each column the occurence of a particular word
data_train = np.zeros((train_df.shape[0], max_words))

i = -1
for idx in train_df.index:
    i += 1
    for word in train_df['clean_text'][idx].split(" "):
        if word in word_idx_map:
            data_train[i][word_idx_map[word]] += 1 # assign values to each datapoint

# SVM classifier model with rbf kernel and class weights balanced
model = SVC(kernel='rbf', gamma='auto', class_weight='balanced')
y_train_actual = np.array(train_df['label_num'])
model.fit(data_train, y_train_actual)
print("Model fitting complete")
print()
print("Results :")

Q1.read_predict_email_svm(word_idx_map, model)