#DIVA.py

import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

class diva:

    def __init__(self, num_channels, input_shape, num_hidden, 
                hidden_act='sigmoid', output_act='linear',
                loss='mean_sqared_error', optimizer=SGD(), 
                prev_model=None, #for putting DIVA at the end of another Keras sequential model, for deep/convolutional DIVAs
                compare=None #What function to determine which reproduction is the most accurate. 
                #Use compareASE for absolute squared error
                ):
        

        self.channels=[]
        self.num_channels=num_channels
        
        if(prev_model==None):
            shared=(Dense(num_hidden, input_shape=(input_shape,)))
            shared_act=(Activation(hidden_act))
            
        if(compare==None):
            self.compare=compareMSE
        else:
            self.compare=compare
        
        
        for x in range(0, num_channels):
            model = Sequential()
            
            if(prev_model==None):
                model.add(shared)
                model.add(shared_act)
                model.add(Dense(input_shape, input_shape=shared_act.output_shape))
            else:
                for layer in prev_model.layers:
                    model.add(layer)
                model.add(Dense(input_shape, input_shape=prev_model.layers[-1].output_shape))
                
            model.add(Activation(output_act))
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
            model.summary()
            self.channels.append(model)

            
    def train(self, X_train, Y_train, nb_epoch, verbose, X_test, Y_test): #train model
    
        losses=[]
        validations=[]
  
        X_train_fixed = batchWorkaround(X_train)
    
        for x in range(0, nb_epoch):
        
            if(verbose==1):
                print('Epoch %d'%(x+1))
            
            loss=0.0
            
            for y in range(0, len(X_train)):
                loss+=self.channels[Y_train[y]].train_on_batch(X_train_fixed[y],X_train_fixed[y])[0]
            
            if(verbose>0):
                print('Loss %f'%loss)
            losses.append(loss)       
 
            validation=(self.test(X_test,Y_test,verbose))
            validations.append(validation)
     
        if(verbose>0):    
            print('Done training')
            
        return ([losses, validations])
    
        
    def raw_predict(self, X_test): # generate raw output from each channel; returns [n_channels][n_samples][n_features]
        
        output=[]
        
        for x in range(0,self.num_channels):
            output.append(self.channels[x].predict(X_test))
        
        return output
    
    
    def predict(self, X_test, verbose): #convert raw output into model predictions
        
        output=self.raw_predict(X_test)

        predictions=[]
        for x in range(0,len(output[0])):
            channel_outs = []
            for y in range(0,self.num_channels):
                channel_outs.append(self.compare(output[y][x],X_test[x]))
            predictions.append(np.argmin(channel_outs))
        
        return predictions
    
    def test(self, X_test, y_test, verbose): #test accuracy on validation data; for checking functionality see diva_tests.py
        
        predictions=self.predict(X_test, verbose)
        
        correct=0.0
        for x in range(0,len(predictions)):
            if(predictions[x]==y_test[x]):
                correct+=1
            
        if(verbose==1):
            print ('Test Accuracy: %f'%(correct/(len(predictions))))    
        
        return (correct/(len(predictions)))   
    
def compareMSE(arrayA, arrayB): #compute Mean Squared Error
    return (np.average(np.square(np.subtract(arrayA, arrayB))))

def compareASE(arrayA, arrayB): #compute Absolute Squared Error
    return (np.sum(np.square(np.subtract(arrayA, arrayB))))

#To workaround the fact that the model backprops from different channels during training, the code trains on every example \
#using a batch size of one, and thus the code expects dimensions of [n_examples][1][n_features] instead of [n_examples][n_features]
def batchWorkaround(train_x): 
    train_x_fixed = []
    for x in range(0,len(train_x)):
        temp=[]
        temp.append(train_x[x])
        temp=np.array(temp)
        train_x_fixed.append(temp)
    return train_x_fixed     
        
#Most ML algorithms expect one-hot output encoding for use with a softmax layer, DIVA does not.
def removeOneHot(train_y): 
    #Implement this later
    print('Not yet implemented')
    assert False