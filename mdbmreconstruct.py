# Implementation of 3 layer ( 2 hidden layer ) Deep Boltzmann Machine
from __future__ import division
import numpy as np
from scipy.special import expit
import pandas as pd
import pickle
#from matplotlib import pylab as plt
from sklearn.metrics import confusion_matrix
def binary_cross_entropy(data, reconst):
    return - np.mean( np.sum( data * np.log(reconst) + (1-data) * np.log(1 - reconst), axis=1) )

'''
def reconstruct_data(data, b, c1, c2, w_vh1, w_h1h2, num_sample=100):
    m_h1 = expit( np.dot(data, w_vh1) + c1 )
    for i in range(num_sample):
        m_h2 = expit( np.dot(m_h1, w_h1h2) + c2 )
        m_h1 = expit( np.dot(w_h1h2, m_h2.T).T + c1 )
    return expit( np.dot(w_vh1, m_h1.T).T + b )
'''

def reconstruct_data(data, b, c1, c2, w_vh1, w_h1h2, w_h2h3,data_right, b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right ,num_sample=100):
    m_h1 = expit(np.dot(data, w_vh1) + c1)
    m_h1_right=expit(np.dot(data_right,w_vh1_right)+c1_right)

    for i in range(num_sample):
        m_h2 = expit( np.dot(m_h1, w_h1h2) + c2 )
        m_h2_right=expit(np.dot(m_h1_right,w_h1h2_right)+c2_right)
        m_h3=expit(np.dot(m_h2,w_h2h3)+np.dot(m_h2_right,w_h2h3_right)+c3)
        m_h2=expit(np.dot(w_h2h3,m_h3.T).T+c2)
        m_h2_right=expit(np.dot(w_h2h3_right,m_h3.T).T+c2_right)
        m_h1 = expit( np.dot(w_h1h2, m_h2.T).T + c1 )
        m_h1_right=expit(np.dot(w_h1h2_right,m_h2_right.T).T+c1_right)
    return expit( np.dot(w_vh1, m_h1.T).T + b ),expit(np.dot(w_vh1_right,m_h1_right.T).T+b_right)

def reconstruct_right_from_left(data, b, c1, c2, w_vh1, w_h1h2, w_h2h3, b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right ,num_sample=100):
    m_h1 = expit(np.dot(data, w_vh1) + c1)
    #m_h1_right=expit(np.dot(data_right,w_vh1_right)+c1_right)

    for i in range(num_sample):
        m_h2 = expit( np.dot(m_h1, w_h1h2) + c2 )
        #m_h2_right=expit(np.dot(m_h1_right,w_h1h2_right)+c2_right)
        m_h3=expit(np.dot(m_h2,w_h2h3)+c3)
        #m_h2=expit(np.dot(w_h2h3,m_h3.T).T+c2)
        m_h2_right=expit(np.dot(w_h2h3_right,m_h3.T).T+c2_right)
        #m_h1 = expit( np.dot(w_h1h2, m_h2.T).T + c1 )
        m_h1_right=expit(np.dot(w_h1h2_right,m_h2_right.T).T+c1_right)
    return expit(np.dot(w_vh1_right,m_h1_right.T).T+b_right)

def reconstruct_left_from_right( b, c1, c2, w_vh1, w_h1h2, w_h2h3,data_right, b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right ,num_sample=100):
        #m_h1 = expit(np.dot(data, w_vh1) + c1)
        m_h1_right = expit(np.dot(data_right, w_vh1_right) + c1_right)

        for i in range(num_sample):
            #m_h2 = expit(np.dot(m_h1, w_h1h2) + c2)
            m_h2_right = expit(np.dot(m_h1_right, w_h1h2_right) + c2_right)
            m_h3 = expit(np.dot(m_h2_right, w_h2h3_right) + c3)
            m_h2 = expit(np.dot(w_h2h3, m_h3.T).T + c2)
            #m_h2_right(np.dot(w_h2h3_right, m_h3.T).T + c2_right)
            m_h1 = expit(np.dot(w_h1h2, m_h2.T).T + c1)
            #m_h1_right = expit(np.dot(w_h1h2_right, m_h2_right.T).T + c1_right)
        return expit(np.dot(w_vh1, m_h1.T).T + b)

def dbm_contrastive_divergence(data, b, c1, c2, w_vh1, w_h1h2, w_h2h3, data_right, b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right,  w_h2h3_right, num_sample=100):
    # Mean field
    m_vis = data
    m_h1 = np.random.uniform(size=(len(data), len(c1)))
    m_h2 = np.random.uniform(size=(len(data), len(c2)))

    m_vis_right=data_right
    m_h1_right=np.random.uniform(size=(len(data_right), len(c1_right)))#len(data) should be = len(data_right)
    m_h2_right = np.random.uniform(size=(len(data_right), len(c2_right)))

    m_h3 = np.random.uniform(size=(len(data),len(c3)))
    for i in range(num_sample):
        m_h1 = expit( np.dot(m_vis, w_vh1) + np.dot(w_h1h2, m_h2.T).T + c1 )
        m_h2 = expit( np.dot(m_h1, w_h1h2) + np.dot(w_h2h3, m_h3.T).T + c2 )

        m_h1_right = expit(np.dot(m_vis_right, w_vh1_right)+np.dot(w_h1h2_right, m_h2_right.T).T+c1_right)
        m_h2_right = expit(np.dot(m_h1_right, w_h1h2_right)+np.dot(w_h2h3_right, m_h3.T).T+c2_right)

        m_h3 = expit(np.dot(m_h2, w_h2h3) + np.dot(m_h2_right, w_h2h3_right) + c3)

    # Gibbs sample
    s_vis = np.random.binomial(1, m_vis)
    s_h1 = np.random.binomial(1, 0.5, size=(len(data), len(c1)))
    s_h2 = np.random.binomial(1, 0.5, size=(len(data), len(c2)))

    s_vis_right=np.random.binomial(1,m_vis_right)
    s_h1_right=np.random.binomial(1,0.5,size=(len(data_right),len(c1_right)))
    s_h2_right=np.random.binomial(1,0.5,size=(len(data_right),len(c2_right)))

    s_h3=np.random.binomial(1,0.5,size=(len(data),len(c3)))

    for i in range(num_sample):
        sm_vis = expit( np.dot(w_vh1, s_h1.T).T + b )
        s_vis = np.random.binomial(1, sm_vis)
        sm_h1 = expit( np.dot(s_vis, w_vh1) + np.dot(w_h1h2, s_h2.T).T + c1 )
        s_h1 = np.random.binomial(1, sm_h1)
        sm_h2 = expit( np.dot(s_h1, w_h1h2) + np.dot(w_h2h3, s_h3.T).T+ c2 )
        #sm_h2 = expit( np.dot(s_h1, w_h1h2) + c2 )
        s_h2 = np.random.binomial(1, sm_h2)

        sm_vis_right=expit(np.dot(w_vh1_right,s_h1_right.T).T+b_right)
        s_vis_right=np.random.binomial(1,sm_vis_right)
        sm_h1_right=expit(np.dot(s_vis_right,w_vh1_right)+np.dot(w_h1h2_right,s_h2_right.T).T+c1_right)
        s_h1_right=np.random.binomial(1,sm_h1_right)
        sm_h2_right=expit(np.dot(s_h1_right,w_h1h2_right)+np.dot(w_h2h3_right,s_h3.T).T+c2_right)
        s_h2_right=np.random.binomial(1,sm_h2_right)

        sm_h3=expit(np.dot(s_h2,w_h2h3)+np.dot(s_h2_right,w_h2h3_right)+c3)
        s_h3=np.random.binomial(1,sm_h3)

    #update_b, update_c1, update_c2, update_w_vh1, update_w_h1h2
    return np.mean(m_vis - s_vis, axis=0), np.mean(m_h1 - s_h1, axis=0), np.mean(m_h2 - s_h2, axis=0), \
                ( np.dot(m_vis.T, m_h1) - np.dot(s_vis.T, s_h1) ) / len(data), ( np.dot(m_h1.T, m_h2) - np.dot(s_h1.T, s_h2) ) / len(data), \
                np.mean(m_vis_right - s_vis_right, axis=0), np.mean(m_h1_right - s_h1_right, axis=0), np.mean(m_h2_right - s_h2_right, axis=0), \
                (np.dot(m_vis_right.T, m_h1_right) - np.dot(s_vis_right.T, s_h1_right)) / len(data), (np.dot(m_h1_right.T, m_h2_right) - np.dot(s_h1_right.T, s_h2_right)) / len(data),\
                np.mean(m_h3-s_h3, axis=0), (np.dot(m_h2.T,m_h3)-np.dot(s_h2.T,s_h3))/len(data),(np.dot(m_h2_right.T,m_h3)-np.dot(s_h2_right.T,s_h3))/len(data)

# Assign structural parameters
num_visible = 78#784
num_hidden1 = 50#500
num_hidden2 = 100#1000

num_visible_right = 21#784
num_hidden1_right = 50#500
num_hidden2_right = 100#1000

num_hidden3=100

# Assign learning parameters
train_epochs = 100
train_learning_rate = 0.01

# Initialize weights
'''
try:
    with open('objs.pickle') as f:  # Python 3: open(..., 'rb')
        b, c1, c2, w_vh1, w_h1h2, w_h2h3, b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right = pickle.load(f)
except FileNotFoundError:
'''
b = np.zeros((num_visible, ))
c1 = np.zeros((num_hidden1, ))
c2 = np.zeros((num_hidden2, ))
w_vh1 = np.random.normal(scale=0.01, size=(num_visible, num_hidden1))
w_h1h2 = np.random.normal(scale=0.01, size=(num_hidden1, num_hidden2))

b_right = np.zeros((num_visible_right, ))
c1_right = np.zeros((num_hidden1_right, ))
c2_right = np.zeros((num_hidden2_right, ))
w_vh1_right = np.random.normal(scale=0.01, size=(num_visible_right, num_hidden1_right))
w_h1h2_right = np.random.normal(scale=0.01, size=(num_hidden1_right, num_hidden2_right))

w_h2h3=np.random.normal(scale=0.01,size=(num_hidden2,num_hidden3))
w_h2h3_right=np.random.normal(scale=0.01,size=(num_hidden2_right,num_hidden3))
c3=np.zeros((num_hidden3, ))


# Load data, data needs to be in range [0, 1]
ddd = pd.read_csv('W:/GWU2.0/ml2/DBM/DeepLearning Data/x_bin_try.csv').iloc[0:10000, 1:]
data = np.array(ddd)

ddd2=pd.read_csv('W:/GWU2.0/ml2/DBM/DeepLearning Data/y_bin_try.csv').iloc[0:10000, 1:]
data_right=np.array(ddd2)

# Fine tuning
for i in range(train_epochs):
    # Calculate gradient
    print(i)
    update_b, update_c1, update_c2, update_w_vh1, update_w_h1h2, \
            update_b_right, update_c1_right, update_c2_right, update_w_vh1_right, update_w_h1h2_right,\
            update_c3, update_w_h2h3,update_w_h2h3_right \
            = dbm_contrastive_divergence(data, b, c1, c2, w_vh1, w_h1h2, w_h2h3, data_right, b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right,  w_h2h3_right)

    #dbm_contrastive_divergence(data, b, c1, c2, w_vh1, w_h1h2, b_right, data_right,c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3,w_h2h3_right, num_sample=100):
    # Update parameters
    b += train_learning_rate * update_b
    c1 += train_learning_rate * update_c1
    c2 += train_learning_rate * update_c2
    w_vh1 += train_learning_rate * update_w_vh1
    w_h1h2 += train_learning_rate * update_w_h1h2
    w_h2h3 += train_learning_rate * update_w_h2h3

    b_right += train_learning_rate * update_b_right
    c1_right += train_learning_rate * update_c1_right
    c2_right += train_learning_rate * update_c2_right
    w_vh1_right += train_learning_rate * update_w_vh1_right
    w_h1h2_right += train_learning_rate * update_w_h1h2_right
    w_h2h3_right += train_learning_rate * update_w_h2h3_right



# Show fine tuning result
#save with pickle
#with open('objs.pickle', 'w') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([b, c1, c2, w_vh1, w_h1h2, w_h2h3,b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right], f)

cost = binary_cross_entropy(data, reconstruct_data(data, b, c1, c2, w_vh1, w_h1h2, w_h2h3,data_right,\
                                                   b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right)[0])\
       +binary_cross_entropy(data_right, reconstruct_data(data, b, c1, c2, w_vh1, w_h1h2, w_h2h3,\
                                                          data_right, b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right)[1])
print( "Reconstruction cost is %.2f"%cost )
#print(reconstruct_data(data, b, c1, c2, w_vh1, w_h1h2))
#print(len(reconstruct_data(data, b, c1, c2, w_vh1, w_h1h2)))

ddd1 = pd.read_csv('W:/GWU2.0/ml2/DBM/DeepLearning Data/x_bin_try.csv').iloc[10000:11000, 1:]
x_left=np.array(ddd1)
recon=reconstruct_right_from_left(x_left,b, c1, c2, w_vh1, w_h1h2, w_h2h3, b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right)
reconround=np.around(recon)
np.savetxt("refromrightfromleft.csv", reconround, delimiter=",")

ddd22=pd.read_csv('W:/GWU2.0/ml2/DBM/DeepLearning Data/y_bin_try.csv').iloc[10000:11000, 1:]
x_right=np.array(ddd22)
recon2=reconstruct_left_from_right(b, c1, c2, w_vh1, w_h1h2, w_h2h3, x_right,b_right, c1_right, c2_right, w_vh1_right, w_h1h2_right, w_h2h3_right)
reconround2=np.around(recon2)
np.savetxt("refromleftfromright.csv", recon2, delimiter=",")

#accuracy for zipqn
def category(recon):
    y = np.empty([0, 0], dtype=int)
    for i in range(len(recon)):#0:99
        for j in range(4):
            k=j+17
            if recon[i][k]==1:
                y=np.append(y,j+1)
    return y

y_true_cat=category(x_right)
y_pred_cat=category(reconround)
print("acc for zipinc")
print(confusion_matrix(y_true_cat, y_pred_cat))

#accuracy for asource
def category2(recon):
    y = np.empty([0, 0], dtype=int)
    for i in range(len(recon)):#0:99
        for j in range(5):
            k=j
            if recon[i][k]==1:
                y=np.append(y,j+1)
    return y

y_true_cat_as=category2(x_right)
y_pred_cat_as=category2(reconround)
print("acc for asource")
print(confusion_matrix(y_true_cat_as, y_pred_cat_as))
'''
#accuracy for atype
def category3(recon):
    y = np.empty([0, 0], dtype=int)
    for i in range(len(recon)):#0:99
        for j in range(6):
            k=j+5
            if recon[i][k]==1:
                y=np.append(y,j+1)
    return y

y_true_cat_at=category3(x_right)
y_pred_cat_at=category3(reconround)
print("acc for atype")
print(confusion_matrix(y_true_cat_at, y_pred_cat_at))
'''
#accuracy for race
def category4(recon):
    y = np.empty([0, 0], dtype=int)
    for i in range(len(recon)):#0:99
        for j in range(6):
            k=j+11
            if recon[i][k]==1:
                y=np.append(y,j+1)
    return y

y_true_cat_r=category4(x_right)
y_pred_cat_r=category4(reconround)
print("acc for race")
print(confusion_matrix(y_true_cat_r, y_pred_cat_r))