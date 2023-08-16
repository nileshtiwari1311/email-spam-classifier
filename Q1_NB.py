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

# find the number of emails in which the word appears
count_words = dict()
for text in train_df['clean_text']:
    words = set(text.split(" "))
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
            data_train[i][word_idx_map[word]] = 1 # assign values to each datapoint

y_train_actual = np.array(train_df['label_num'])

count_0 = [0 for i in range(max_words)] # for each word, number of non-spam e-mails containing that word
count_1 = [0 for i in range(max_words)] # for each word, number of spam e-mails containing that word
n_0 = 0 # number of non-spam emails
n_1 = 0 # number of spam emails

# find the values of count_0 count_1 list
for i in range(data_train.shape[0]) :
    if y_train_actual[i] == 0 :
        n_0 += 1
        for j in range(data_train.shape[1]) :
            count_0[j] += data_train[i][j]
    else :
        n_1 += 1
        for j in range(data_train.shape[1]) :
            count_1[j] += data_train[i][j]

# find the respective probabilities of each word for spam and non-spam
prob_0 = [0 for i in range(max_words)]
prob_1 = [0 for i in range(max_words)]

for i in range(max_words) :
    prob_0[i] = (1 + count_0[i]) / (2 + n_0) # using Laplacian smoothing
    prob_1[i] = (1 + count_1[i]) / (2 + n_1) # using Laplacian smoothing

prior_0 = n_0 / (n_0 + n_1) # finding the prior for spam
prior_1 = 1 - prior_0

# first find the predicted labels for the train data set
y_train_pred = []

# using the log of the probability as the product of probability values can cause underflow
log_prob_0 = np.log(prior_0)
log_prob_1 = np.log(prior_1)

for text in train_df['clean_text']:
    words = set(text.split(" "))
    for word in word_idx_map:
        if word in words : # if the word is present in the train dataset
            idx = word_idx_map[word]
            log_prob_0 += np.log(prob_0[idx])
            log_prob_1 += np.log(prob_1[idx])
        else : # if the word is not present in the train dataset
            idx = word_idx_map[word]
            log_prob_0 += np.log(1 - prob_0[idx])
            log_prob_1 += np.log(1 - prob_1[idx])
    if log_prob_1 > log_prob_0 : # assign label=1 when prob(y=1|x) > prob(y=0|x)
        y_train_pred.append(1)
    else :
        y_train_pred.append(0)

y_train_pred = np.array(y_train_pred)
accuracy = (y_train_actual == y_train_pred).sum() / len(y_train_actual) * 100
print("Accuracy on train data set =", accuracy)

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
plt.title('Confusion Matrix for Train Data Set (NB)')
plt.savefig('Fig_Q1_NB_train_cm.png', dpi=200, bbox_inches='tight')
plt.close()

# find the predicted labels for test dateset
test_df['clean_text'] = test_df['text'].apply(Q1.text_preprocess)
y_test_actual = np.array(test_df['label_num'])

y_test_pred = []

log_prob_0 = np.log(prior_0)
log_prob_1 = np.log(prior_1)

for text in test_df['clean_text']:
    words = set(text.split(" "))
    for word in word_idx_map:
        if word in words :
            idx = word_idx_map[word]
            log_prob_0 += np.log(prob_0[idx])
            log_prob_1 += np.log(prob_1[idx])
        else :
            idx = word_idx_map[word]
            log_prob_0 += np.log(1 - prob_0[idx])
            log_prob_1 += np.log(1 - prob_1[idx])
    if log_prob_1 > log_prob_0 :
        y_test_pred.append(1)
    else :
        y_test_pred.append(0)

y_test_pred = np.array(y_test_pred)
accuracy = (y_test_actual == y_test_pred).sum() / len(y_test_actual) * 100
print("Accuracy on test data set =", accuracy)

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
plt.title('Confusion Matrix for Test Data Set (NB)')
plt.savefig('Fig_Q1_NB_test_cm.png', dpi=200, bbox_inches='tight')
plt.close()