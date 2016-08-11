#Code functionality tests

#TODO: add tests for deep and convolutional DIVA

import DIVA

def channelTest(model): # testing if the shared layer is actually shared, and that the channels are truly independent
    
    print('Testing shared layer')
    for x in model.channels:
        assert id(model.channels[0].layers[-4])==id(x.layers[-4]) # shared weights
        assert id(model.channels[0].layers[-3])==id(x.layers[-3]) # shared activation
        
    print('Testing divergent channels')
    for x in range(0, len(model.channels)):
        for y in range(0, len(model.channels)):
            if(x==y):
                continue
            assert model.channels[x].layers[-2]!=model.channels[y].layers[-2] # divergent weights
            assert model.channels[x].layers[-1]!=model.channels[y].layers[-1] # divergent activation
            
    print('Done testing')

def testLearning(diva_model, X_train, y_train, X_test, y_test): # run two epochs and confirm that the model learned
    train_metrics = diva_model.train(X_train, y_train, 1, 1, X_test, y_test)            
    accuracy = diva_model.test(X_test, y_test, 1)
    
    train_metrics_2 = diva_model.train(X_train, y_train, 1, 1, X_test, y_test)            
    accuracy_2 = diva_model.test(X_test, y_test, 1)
    
    print('Checking for loss reduction')
    assert(train_metrics > train_metrics_2)
    
    print('Checking for test accuracy increase')
    assert(accuracy < accuracy_2)
    
    print('Done testing')
    
def testDiva():
    import numpy as np
    np.random.seed(1337)  # for reproducibility

    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.utils import np_utils

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    nb_classes = 10
    nb_epoch = 2
    num_channels = 10
    input_shape = 784
    num_hidden = 500

    diva_model = DIVA.diva(nb_classes, input_shape, num_hidden, hidden_act='relu', optimizer=RMSprop())
    
    print('Model compiled')
    
    channelTest(diva_model)
    
    print('Testing learning')
    testLearning(diva_model, X_train, y_train, X_test, y_test)
    
    print('Restesting channels')
    channelTest(diva_model)

    print('Done')
    
if(__name__ == '__main__'):
    testDiva()
    
