README

---- Intro ----

This is a Keras, Python & Numpy based implementation of the Divergent Autoencoder(DIVA) model of human category learning (Kurtz, 2007). There is also demo code to run it on the MNIST dataset as a machine learning algorithm. 

The purpose of this was to determine if DIVA could be compared to other machine learning algorithms, as well as performance on a much larger dataset than had been previously tested. This is a refactored version of the original proof of concept script using Keras to utilize a gpu via Theano and CUDA. 

---- Background	----

The DIVA model is a classifier that consists of n backpropagation network autoencoders, known as channels, where n is the number of classes. All the channels share a hidden weight layer, but have their own output layers. Each is trained on the members of a single class, though the hidden layer is exposed to all classes as it is shared between all the channels. During classification, the input is run through every channel, and the one with the lowest reproduction error is what the model believes is the correct class.

The implementation uses multiple autoencoders, one for each class, that all share a hidden layer. It then trains per example (i.e. batch size=1) on the corresponding channel to work around the fact that most frameworks assume backpropagation from every output. This prevents the use of batch training. Though thematic consistency has been maintained wherever possible, the nature of the workaround requires some departure from how base Keras normally operates.

---- Function defs ----
#After importing DIVA.py, create a new diva object, and use the train, predict, and test functions. 

#Initialize object. Uses Keras defaults for weight initialization
def __init__(self, num_channels, input_dim, num_hidden, 	# int, int, int. 
	hidden_act='sigmoid', 					#hidden layer activation, cf https://keras.io/activations/
	loss='mean_sqared_error', 				#Loss function while training model. cf https://keras.io/objectives/
	optimizer=SGD(), 					#Stochastic gradient descent. More advanced optimizers don't seem to work well, presumably because the expected loss surface changes between examples
	prev_model=None, 					#Attach DIVA to the end of another Keras sequential model, instead of making a new shared layer use this for deep/convolutional/dropout DIVAs
	compare=None 						#What function to determine which reproduction is the most accurate, Defaults to compareMSE for mean squared error, use compareMAE for mean absolute error
	):
        
#All functions expect numerical feature data in the form X_data[n_examples][n_features] and the correct channel number at Y_data[n_examples]
#alt_X_train and alt_X_test is for Expected output if different from input. Needed for convolution since Keras expects a different, preprocessed shape.

def train(self, X_train, Y_train, nb_epoch, 			#Train model. Set verbose to 1 to print epoch, loss, and validation accuracy on every epoch, zero to suppress.
		verbose, X_test, Y_test				#Returns [losses[n_epochs],validation_accuracy[n_epochs]]
		alt_X_train=None, alt_X_test=None):		
 			
def predict(self, X_test, alt_X_test=None):	 		#Convert raw output into model predictions; returns a list of predictions with the same length as the test data.
  
def test(self, X_test, y_test,  				#Returns test accuracy on validation data; for checking functionality see diva_tests.py 
verbose, alt_X_test=None):					#Set verbose to 1 to print test accuracy

---- Files ----

DIVA_MNIST.ipynb runs DIVA on MNIST
DIVA_CNN.ipynb compares convolutional MLP to convolutional DIVA
DIVA_Deep.ipynb compares deep MLP to deep DIVA
DIVA_Starved.ipynb compares MLP and DIVA when the number of hidden units is extremely restricted

DIVA.py contains the core code

diva_tests.py contains code tests
DIVA_Tests.ipynb is for running these tests in jupyter easily


---- References ----

Original DIVA paper:
	Kurtz, K. J. (2007). The divergent autoencoder (DIVA) model of category learning. Psychonomic Bulletin & Review, 14(4), 560-576.

