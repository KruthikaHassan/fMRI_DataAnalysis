from data_fMRI import *
from shooting_algorithm import *


def main():

    ################################## Load the X_train, jth voxel corresponding to ith trial word ################################

    X_train = Load_Data_Mtx('subject1_fmri_std.train.mtx')

    ################################# Load Y_train consisting of 218 semantic feature vector ######################################

    word_id_train = Load_csv('subject1_wordid.train.csv')
    word_feature = Load_Data_Mtx('word_feature_centered.mtx')
    Y_train = label_vector(word_id_train, word_feature)

    ################################# Implement coordinate descent using shooting algorithm ########################################

    X = np.matrix(X_train)
    Y = np.matrix(Y_train)
    lambda = 0.05
    beta = shooting(X,Y,lambda)

    return beta

#######################################################################################################################################

beta = main()
print(beta)