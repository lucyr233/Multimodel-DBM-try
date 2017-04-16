import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import random
#import csv
from numpy import genfromtxt

my_data=genfromtxt("W:/GWU2.0/ml2/DBM/DeepLearning Data/data_c30.csv",
                   delimiter=',')

origin=pd.read_csv("W:/GWU2.0/ml2/DBM/DeepLearning Data/data_c30.csv")

wondercol=[4-1,5-1,76-1,77-1,78-1]

#train_left=origin.iloc[:,-wondercol]
train_right=origin.iloc[:,wondercol]
my_data_train_right=my_data[:,wondercol]
my_data_train_left=np.delete(my_data,wondercol,axis=1)
train_left=origin.drop(origin.columns[[3, 4, 75,76,77]], axis=1)
len(train_left)
#random.randint(0,107)
selectfeature=random.sample(range(0,107),50)
validation_left=
validation_right=

# trying from what i understand
# v_left to h_left to h_join to h_right to v_right

def from_down_to_up(self, down)  # hidden from visible
    #gaussian-berenouli
    #upprobGB=tf.nn.sigmoid(tf.matmul(down/sigma,self.W)+self.biasup)
    #replicate-softmax
    #upprobSM=tf.nn.sigmoid(M*biasup+tf.matmul(down,self.W))
    #binary, from visible to hidden it does not matter becasue hidden always binary
    upprob = tf.nn.sigmoid(
        tf.matmul(down, self.W) + self.biasup)  # hprobs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.bh_)
    return upprob #, upstates


def from_up_to_down(self, up)  # visible from hidden
    #gaussian-berenouli
    downprobGB=scipy.stats.norm(biasdown+sigma*tf.matmul(up, tf.transpose(self.W)),sigma_2, 1).pdf(0)
    #replicate-softmax
    downprobSM=exp(biasdown+tf.matmul(up, tf.transpose(self.W)))/sum(exp(biasdown+tf.matmul(up, tf.transpose(self.W))))
    downprob = tf.nn.sigmoid(tf.matmul(up, tf.transpose(self.W)) + self.biasdown)
    # visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv
    # vprobs = tf.nn.sigmoid(visible_activation)#bis=bin
    return downprob


def updateWandbias(self, down, )  # inputdata,hprobs0,hstates0,vprobs,hprobs1,learningrate
    postive =
    # =tf.matmul(tf.transpose(visible), hidden_states)#vis=bin
    negative =
    # =tf.matmul(tf.transpose(vprobs), hprobs1)
    W_update =
    # =self.W.assign_add(self.learning_rate * (positive - negative))
    biasup_update =
    # =self.bh_.assign_add(self.learning_rate * tf.reduce_mean(hprobs0 - hprobs1, 0))
    biasdown_update =
    # =self.bv_.assign_add(self.learning_rate * tf.reduce_mean(self.input_data - vprobs, 0))

    return W_update, biasup_update, biasdown_update


v_left = train_left[:,selectfeature]
v_right = train_right

self.W=tf.Variable(tf.random_normal(self.num_visible,self.num_hidden),mean=0.0,stddev=0.01,name='weights')
self.biasup=tf.Variable(tf.Zeros([self.num_hidden]),name='biasup')
self.biasdown=tf.Variable(tf.Zeros([self.num_visible]),name='biasdown')

W= np.random.normal(0.0,0.01,50)
biasup=[0]*len(v_left)
biasdown=[0]*len(v_left)

upprob = tf.nn.sigmoid(
        tf.matmul(down, self.W) + self.biasup)  # hprobs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.bh_)
up=np.matmul(v_left,W)+biasup
upprob=1/(1+math.exp(-up))


h_left = from_down_to_up(v_left)
h_right = from_down_to_up(v_right)

hpcom = h_left.extend(h_right)

h_join = from_down_to_up(hpcom)
hpcom2 = from_uo_to_down(h_join)
h_join2 = from_down_to_up(hpcom2)

WJ, biasupJ, biasdownJ = updateWandbias()

h_left1, h_right1 = hpcom2[1:50], hpcom2[51:55]

v_left1 = down_from_up(h_left1)
v_right1 = down_from_up(h_right1)

W_left, biasup_left, biasdown_left = updateWandbias(v_left, h_left, v_left1, h_left1)
W_right, biasup_right, biasdown_right = updateWandbias(v_right, h_right, v_right1, h_right1)

# then iterate with v_left1 and v_right1

err=V_left-v_left1
err_sum=tf.reduce_mean(err*err)
self.loss_function=tf.sqrt(tf.reduce_mean(tf.square(V_left-V_left1)))
