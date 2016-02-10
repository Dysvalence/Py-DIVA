README

Intro 
	This is a python & numpy based implementation of the Divergent Autoencoder(DIVA) model of human category learning (Kurtz, 2007), configured to run on the MNIST dataset as a machine learning algorithm. 

	The purpose of this was to determine if DIVA could be compared to other machine learning algorithms, as well as performance on a much larger dataset than had been previously tested. Early results are promising, with 80% accuracy within a few epochs, though the current implementation is too slow to allow for full training. 

	As such development has halted in favor of a python and theano implementation that can utilize a GPU for higher speeds. Since is defunct academic testing code and not designed to be maintainable production code, large amounts of messy commented code has been left behind to document the development and testing process in lieu of redirecting efforts from the GPU version for cleanup.

Background	
	The DIVA model is a classifier that consists of n backpropagation network autoencoders, known as channels, where n is the number of classes. All the channels share a hidden weight layer, but have their own output layers. Each is trained on the members of a single class, though the hidden layer is exposed to all classes as it is shared between all the channels. During classification, the input is run through every channel, and the one with the lowest reproduction error is what the model believes is the correct class.

	
Files
	The project should have been divided across multiple files; A text search for the following keywords will find the points at which I would have split the code:
	
	running MNIST section
	DIVA class section
	other functions section
	data loading section
	
	The file itself is DIVA_V7.py
	
	
Usage
	Running DIVA.py will setup the environment. Use test(lastRun=None, epochs=1) to run the code; lastRun can be given a 10 element list of diva objects and it will resume training/testing. Set epochs to the desired number or to zero for running the test set only. Other useful analysis functions are explained in comments.

Credits
	Dataset loading code, print statement format, general program flow and much of the parameters was heavily based on MILA lab code:
    https://github.com/lisa-lab/DeepLearningTutorials
	
	More information:
	http://deeplearning.net/tutorial/gettingstarted.html

	Direct link to dataset file:
    http://deeplearning.net/data/mnist/mnist.pkl.gz

	Original DIVA paper:
	Kurtz, K. J. (2007). The divergent autoencoder (DIVA) model of category learning. Psychonomic Bulletin & Review, 14(4), 560-576.
