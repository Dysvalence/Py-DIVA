README

---- Intro ----

This is a Keras, Python & Numpy based implementation of the Divergent Autoencoder(DIVA) model of human category learning (Kurtz, 2007), configured to run on the MNIST dataset as a machine learning algorithm. 

The purpose of this was to determine if DIVA could be compared to other machine learning algorithms, as well as performance on a much larger dataset than had been previously tested. This is a proof of concept script using Keras to utilize a gpu via Theano and CUDA. It's incredibly hacky, academic code and I planned to refactor it before even starting, since it was not guaranteed that the implementation I used would even work with Keras; early attempts with Theano were erratic due to caching.

---- Background	----

The DIVA model is a classifier that consists of n backpropagation network autoencoders, known as channels, where n is the number of classes. All the channels share a hidden weight layer, but have their own output layers. Each is trained on the members of a single class, though the hidden layer is exposed to all classes as it is shared between all the channels. During classification, the input is run through every channel, and the one with the lowest reproduction error is what the model believes is the correct class.

The implementation uses 10 autoencoders that share a hidden layer, and trains per example (i.e. batch size=1) on the corresponding channel to work around the fact that most frameworks assume backpropagation from every output.

	
---- Files ----

The script is in the DIVA.ipnyb jupyter notebook. Standalone packages are in the works

---- References ----
Original DIVA paper:
	Kurtz, K. J. (2007). The divergent autoencoder (DIVA) model of category learning. Psychonomic Bulletin & Review, 14(4), 560-576.

