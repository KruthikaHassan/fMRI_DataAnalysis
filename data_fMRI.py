import csv
import scipy as sp 
import numpy as np 
import time
from scipy.io import loadmat

################################ Constants, train data, test data, trials ###################################

voxels = 21764         #dimension of input MRI feature
semantic = 218         #dimension of output semantic feature
train_ex = 300         #number of training examples
test_ex = 60           #number of test examples


################################  function to  Load .mtx files     ###########################################

def Load_Data_Mtx(filename):
    data = sp.io.mmread(filename)
    return data 

###############################   function to load csv files ###########################################

def Load_csv(filename):
    file = open(filename)
    csv_data = [row for row in csv.reader(file)]

    return csv_data 

################################ function to create Y_train consisting of semantic labels ##############################################

def label_vector(word_id,feature_vector):
    Y_train=np.zeros((semantic,train_ex))
    wordid_list = []
    feature_transpose = np.transpose(feature_vector)

    for ind in range(0,len(word_id)):
        val = int(word_id[ind][0])
        wordid_list.append(val)

    
    for i in range(0,semantic):
        for j in range(0,train_ex):
            word_index= wordid_list[j]
            Y_train[i][j] = feature_transpose[i][word_index-1]

    return Y_train