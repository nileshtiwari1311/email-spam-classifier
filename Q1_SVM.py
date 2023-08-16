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
print("Size of the dataset =", df.shape)
print()


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

y_train_actual = np.array(train_df['label_num'])

# SVM classifier model with rbf kernel and class weights balanced
model = SVC(kernel='rbf', gamma='auto', class_weight='balanced')
model.fit(data_train, y_train_actual)

# first find the predicted labels for the train data set
y_train_pred = model.predict(data_train)
y_train_pred = np.array(y_train_pred)
print("Accuracy on train data set =",(y_train_actual == y_train_pred).sum() / len(y_train_actual) * 100)

arr = Q1.my_confusion_matrix(y_train_actual, y_train_pred)
print("Confusion Matrix on train data set :")
print(arr)

tn = arr[0][0] # true negative
fp = arr[0][1] # false positive
fn = arr[1][0] # false negative
tp = arr[1][1] # true positive

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Precision on train data set =", precision)
print("Recall on train data set =", recall)
print()

ax = sns.heatmap(arr, annot=True, fmt='g')
ax.set(xlabel='Predicted Label', ylabel='True Label')
plt.title('Confusion Matrix for Train Data Set (SVM)')
plt.savefig('Fig_Q1_SVM_train_cm.png', dpi=200, bbox_inches='tight')
plt.close()

test_df['clean_text'] = test_df['text'].apply(Q1.text_preprocess)

# find the predicted labels for test dateset
y_test_actual = np.array(test_df['label_num'])

data_test = np.zeros((test_df.shape[0], max_words))

i = -1
for idx in test_df.index:
    i += 1
    for word in test_df['clean_text'][idx].split(" "):
        if word in word_idx_map:
            data_test[i][word_idx_map[word]] += 1

y_test_pred = model.predict(data_test)

y_test_pred = np.array(y_test_pred)
print("Accuracy on test data set =",(y_test_actual == y_test_pred).sum() / len(y_test_actual) * 100)

arr = Q1.my_confusion_matrix(y_test_actual, y_test_pred)
print("Confusion Matrix on test data set :")
print(arr)

tn = arr[0][0]
fp = arr[0][1]
fn = arr[1][0]
tp = arr[1][1]

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Precision on test data set =", precision)
print("Recall on test data set =", recall)
print()

ax = sns.heatmap(arr, annot=True, fmt='g')
ax.set(xlabel='Predicted Label', ylabel='True Label')
plt.title('Confusion Matrix for Test Data Set (SVM)')
plt.savefig('Fig_Q1_SVM_test_cm.png', dpi=200, bbox_inches='tight')
plt.close()