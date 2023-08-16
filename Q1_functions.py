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


# preprocessing data
def text_preprocess(text) :
    text = text.lower() # make all letters in lowercase
    text = re.sub(r"https?://\S+", "", text) # remove links
    text = re.sub(r"www\.\S+", "", text) # remove links
    text = re.sub(r"<.*?>", "", text) # remove html tags
    text = "".join([i for i in text if i not in string.punctuation]) # remove punctuation marks
    text = "".join([i for i in text if not i.isdigit()]) # remove digits
    text = " ".join(text.split())
    return text


# list of indices of the complete dataset divided into train and test
def train_test_idx_list(df) :
    test_size = df.shape[0] // 5 # test_size = 0.2
    idx_list = [i for i in range(df.shape[0])]
    test_idx_list = sample(idx_list, test_size) # 20% indices in test data
    train_idx_list = list(set(idx_list).difference(test_idx_list)) # 80% indices in train data
    return train_idx_list, test_idx_list


# divide dataset into train and test dataset
def train_test_split(df) :
    # split the dataset into train and test set with train set = 80%
    df_1 = df[df['label_num']==1] # separate data with label=1
    df_0 = df[df['label_num']==0] # separate data with label=0
    train_idx_list_1, test_idx_list_1 = train_test_idx_list(df_1) # list of indices for train and test data set with label=1
    train_idx_list_0, test_idx_list_0 = train_test_idx_list(df_0) # list of indices for train and test data set with label=0
    train_df_1 = df_1.iloc[train_idx_list_1] # extract train dataset with label=1 using the indices obtained
    test_df_1 = df_1.iloc[test_idx_list_1] # extract test dataset with label=1 using the indices obtained
    train_df_0 = df_0.iloc[train_idx_list_0] # extract train dataset with label=0 using the indices obtained
    test_df_0 = df_0.iloc[test_idx_list_0] # extract test dataset with label=0 using the indices obtained
    train_df = pd.concat([train_df_1, train_df_0]) # merge the train datasets with different labels
    test_df = pd.concat([test_df_1, test_df_0]) # merge the test datasets with different labels
    return train_df, test_df


# return confusion matrix
def my_confusion_matrix(y_actual, y_pred) :
    arr = np.zeros((2,2)).astype(int).reshape(2,2)

    for i in range(len(y_actual)) :
        a = y_actual[i]
        b = y_pred[i]
        if a == b : # true positive/true negative
            arr[a][a] = arr[a][a] + 1
        else : # false positive/false negative
            arr[a][b] = arr[a][b] + 1
    return arr

# predict label for the given email using svm classifier
def read_predict_email_svm(word_idx_map, model) :
    for filename in glob.glob('test/email*.txt') :
        with open(filename, 'r') as f:
            data = f.read() # read the contents of file 'email#.txt'
            text = text_preprocess(data) # preprocess the data
            f.close()
            print('Filename =', os.path.basename(filename))
            
            y_pred = 0
            # convert the data into format required for classification
            data_test = np.zeros((1, len(word_idx_map)))
            for word in text.split(" ") :
                if word in word_idx_map:
                    data_test[0][word_idx_map[word]] += 1
            
            # predict the label using SVM classifier model
            y_pred = model.predict(data_test)
            y_pred = y_pred[0]
            print("Prediction =", y_pred)
            print()