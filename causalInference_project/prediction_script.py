# import dependencies

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# tf.keras.callbacks.Callback()
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import SGD

from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model


# function for loading the data
def load_IHDP_data(training_data,testing_data,i=7):
    with open(training_data,'rb') as trf, open(testing_data,'rb') as tef:
        train_data=np.load(trf); test_data=np.load(tef)
        y=np.concatenate((train_data['yf'][:,i], test_data['yf'][:,i])).astype('float32') #most GPUs only compute 32-bit floats
        t=np.concatenate((train_data['t'][:,i], test_data['t'][:,i])).astype('float32')
        x=np.concatenate((train_data['x'][:,:,i], test_data['x'][:,:,i]),axis=0).astype('float32')
        mu_0=np.concatenate((train_data['mu0'][:,i], test_data['mu0'][:,i])).astype('float32')
        mu_1=np.concatenate((train_data['mu1'][:,i], test_data['mu1'][:,i])).astype('float32')
 
        data={'x':x,'t':t,'y':y,'t':t,'mu_0':mu_0,'mu_1':mu_1}
        data['t']=data['t'].reshape(-1,1) #we're just padding one dimensional vectors with an additional dimension 
        data['y']=data['y'].reshape(-1,1)
        
        #rescaling y between 0 and 1 often makes training of DL regressors easier
        data['y_scaler'] = StandardScaler().fit(data['y'])
        data['ys'] = data['y_scaler'].transform(data['y'])
    return data

data = load_IHDP_data(training_data='./ihdp_npci_1-100.train.npz',testing_data='./ihdp_npci_1-100.test.npz')


# loading the save model

model_weight_dir = 'Weights'
os.makedirs(model_weight_dir, exist_ok = True)
tarnet_load_model = tf.keras.models.load_model(os.path.join(model_weight_dir,'tarnet_model.tf'), compile=False) 
# print(tarnet_load_model.summary())


# Since we know the true CATEs in our simulations, let's plot our results against the true values..
def plot_cates(y0_pred,y1_pred,data):
  #dont forget to rescale the outcome before estimation!
  y0_pred = data['y_scaler'].inverse_transform(y0_pred)
  y1_pred = data['y_scaler'].inverse_transform(y1_pred)
  cate_pred=(y1_pred-y0_pred).squeeze()
  cate_true=(data['mu_1']-data['mu_0']).squeeze() #Hill's noiseless true values
  ate_pred=tf.reduce_mean(cate_pred)

  print(pd.Series(cate_pred).plot.kde(color='blue'))
  print(pd.Series(cate_true).plot.kde(color='green'))

  print(pd.Series(cate_true-cate_pred).plot.kde(color='red'))
  pehe=tf.reduce_mean( tf.square( ( cate_true - cate_pred) ) )
  sqrt_pehe=tf.sqrt(pehe).numpy()
  print("\nSQRT PEHE:",sqrt_pehe)
  print("Estimated ATE (True is 4):", ate_pred.numpy(),'\n\n')
  
  print("\nError CATE Estimates: RED")
  print("Individualized CATE Estimates: BLUE")
  print("Individualized CATE True: GREEN")
  
  return sqrt_pehe,np.abs(ate_pred.numpy()-4)


# Estimating the ATE/CATE
concat_pred = tarnet_load_model.predict(data['x'])
y0_pred_tarnet,y1_pred_tarnet = (concat_pred[:, 0].reshape(-1,1)),(concat_pred[:, 1].reshape(-1,1))
print(plot_cates(y0_pred_tarnet,y1_pred_tarnet,data))