#DIVA.py

"""
DIVA only backpropagates a training example on the corresponding channel, but most ML libraries expect backpropagation from every output. 
To work around this, the diva class contains n seperate models for n classes, but they all share the same hidden layer. 
During training each example is backpropagated only on the model for the corresponding channel. 
Method diva.train() literally does this by setting the batch size to 1, so batch training is not an option.
This caused caching issues the last time I tried doing this in Theano, so I expect this to be a bottleneck, but there aren't any obvious solutions to this.
"""


import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

class diva:

    #sets up the model
    def __init__(self, num_channels, input_dim, num_hidden, # int, int, int. 
                hidden_act='sigmoid', #hidden layer activation, cf https://keras.io/activations/
                loss='mean_sqared_error', #Loss function while training model. cf https://keras.io/objectives/
                optimizer=SGD(), #Stochastic gradient descent. More advanced optimizers don't seem to work well 
                prev_model=None, #Attach DIVA to the end of another Keras sequential model, instead of making a new shared layer
                #Use this for deep/convolutional/dropout DIVAs
                compare=None #What function to determine which reproduction is the most accurate.
                #Defaults to compareMSE for mean squared error, use compareMAE for mean absolute error
                ):
        

        self.channels=[]
        self.num_channels=num_channels
        
        if(prev_model is None):
            shared=(Dense(num_hidden, input_shape=(input_dim,)))
            shared_act=(Activation(hidden_act))
        
        if(compare is None):
            self.compare=compareMSE
        else:
            self.compare=compare
        
        for x in range(0, num_channels):
            model = Sequential()
            
            if(prev_model is None):
                model.add(shared)
                model.add(shared_act)
                model.add(Dense(input_dim, input_shape=shared_act.output_shape))
            else:
                for layer in prev_model.layers:
                    model.add(layer)
                model.add(Dense(input_dim, input_shape=prev_model.layers[-1].output_shape))
                
            model.add(Activation('linear'))
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
            #model.summary()
            self.channels.append(model)

            
    def train(self, X_train, Y_train, nb_epoch, verbose, X_test, Y_test, alt_X_train=None, alt_X_test=None): #train model
        #alt_X_train is for situations when the input and expected output are different due to preprocessing, 
        #Notably this includes convolution- keras expects slightly different input shapes for that. cf MNIST_CNN.ipynb for an example.
        
        losses=[]
        validations=[]
  
        X_train_fixed = batchWorkaround(X_train)
        if(alt_X_train is not None):
            alt_X_train_fixed = batchWorkaround(alt_X_train)
            
        for x in range(0, nb_epoch):
        
            if(verbose>0):
                print('Epoch %d'%(x+1))
            
            loss=0.0
            
            for y in range(0, len(X_train)):
                if(alt_X_train is None):
                    loss+=self.channels[Y_train[y]].train_on_batch(X_train_fixed[y],X_train_fixed[y])[0]
                else:
                    loss+=self.channels[Y_train[y]].train_on_batch(X_train_fixed[y],alt_X_train_fixed[y])[0]

            if(verbose>0):
                print('Loss %f'%loss)
            losses.append(loss)       
 
            if(alt_X_test is None):
                validation=(self.test(X_test,Y_test,verbose))
            else:
                validation=(self.test(X_test,Y_test,verbose, alt_X_test=alt_X_test))
            validations.append(validation)
     
        if(verbose>0):    
            print('Done training')
            
        return ([losses, validations])
    
        
    def raw_predict(self, X_test): # generate raw output from each channel; returns [n_channels][n_samples][n_features]
        
        output=[]
        
        for x in range(0,self.num_channels):
            output.append(self.channels[x].predict(X_test))
        
        return output
    
    
    def predict(self, X_test, alt_X_test=None): #convert raw output into model predictions. alt_X_test is for when input and output differ due to preprocessing, cf comments in train()
        
        output=self.raw_predict(X_test)

        predictions=[]
        for x in range(0,len(output[0])):
            channel_outs = []
            for y in range(0,self.num_channels):
                if(alt_X_test is None):
                    channel_outs.append(self.compare(output[y][x],X_test[x]))
                else:
                    channel_outs.append(self.compare(output[y][x],alt_X_test[x]))                    
            predictions.append(np.argmin(channel_outs))
        
        return predictions
    
    def test(self, X_test, y_test, verbose, alt_X_test=None): #test accuracy on validation data; for checking functionality see diva_tests.py 
        #alt_X_test is for when input and output differ due to preprocessing, cf comments in train()
        
        predictions=self.predict(X_test, alt_X_test=alt_X_test)
        
        correct=0.0
        for x in range(0,len(predictions)):
            if(predictions[x]==y_test[x]):
                correct+=1
            
        if(verbose>0):
            print ('Test Accuracy: %f'%(correct/(len(predictions))))    
        
        return (correct/(len(predictions)))   
    
    
def compareMSE(arrayA, arrayB): #compute Mean Squared Error
    return (np.average(np.square(np.subtract(arrayA, arrayB))))

def compareMAE(arrayA, arrayB): #compute Mean Absolute Error
    return (np.average(np.absolute(np.subtract(arrayA, arrayB))))

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
        
"""
TODO: finish this
#Most ML algorithms expect one-hot output encoding for use with a softmax layer, DIVA does not.
def removeOneHot(train_y): 
    #Implement this later
    print('Not yet implemented')
    assert False
"""