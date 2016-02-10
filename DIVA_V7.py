# -*- coding: utf-8 -*-
"""
Running MNIST
"""

import cPickle
import gzip
import os
import timeit

import numpy
from copy import deepcopy

import scipy
try:
    import PIL.Image as Image
except ImportError:
    import Image

#global variables are deliberately used to allow resuming after keyboard interrupts and code changes.

#holds the latest model
try:
    lastModel
except NameError:
    lastModel = 0 


#holds the best model based on validation data
try:
    bestModel
except NameError:
    bestModel = 0 


#holds the outputs generated during testing. Rearranged for easier variance analysis.

try:
    similarityData
except NameError:
    similarityData = 0 


#if(lastR == None):
#lastR = 0 
#if(bestModel==None):
bestModel = ''
#if(rsl==None):
dataset = 0
#if(lastModel==None):
lastModel = ''


def test(lastRun=None, epochs=1):
    print '...setup'
    global bestModel
    global lastModel
    global lastR
    global lastV
    global dataset

    dataset = load_data('mnist.pkl.gz')
        
    rng = numpy.random.RandomState(1234)  

    if(lastRun==None):

        sharedWeights=numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (500+785)),
            high=numpy.sqrt(6. / (500+785)),
            size=(500, 785)
            ),
            dtype=numpy.float32)
#sharedBiases=numpy.random.rand(500)

        divas = []


        for x in range(0,10):
            divas.append(
                diva(
                    sharedWeights,
                    rng            
                    )      
            )    
    else:
        divas=lastRun
    
    start_time = timeit.default_timer()
    print '...training'
    learning_rate=.5
    pastCorrect=0
    
    similarityCheck = []
    
    for e in range(0, epochs):
        epoch_time = timeit.default_timer()
        print 'epoch %d' % (e)
        print 'training'
        #for s in range(0, 49999):
        meanError=[]
        for r in range(0, 100):
            err=[]
            for s in range((r*500), (500*(r+1))):
                channel=int(dataset[0][1][s])        
                err.append(divas[channel].train(dataset[0][0][s],learning_rate))
            print numpy.mean(err)
            meanError.append(err)            
            #if(learning_rate>.001):
               #learning_rate-=.001
               #print 'Learning rate dropping to %f' % (learning_rate)
        print ('Mean error: %f'%(numpy.mean(meanError)))
        
        print "...validating"
        totalcorrect=0 
        for k in range(0,20):
            correct=0 
            for s in range((k*500),((k+1)*500)):
                outs = []
                for c in range(0, 10):
                    outs.append(proc(divas[c],dataset[1][0][s]))
                pred=outs.index(min(outs))
                act=int(dataset[1][1][s])
                if(pred==act):
                    correct+=1
                #similarityCheck.append(outs)
                #print 'pred: %d actual: %d' % (pred, act)
            print 'batch validation %d of 500' % (correct)
            
            if(correct<pastCorrect):
                correct=correct
                #if(learning_rate>.001):
                #    learning_rate*=.8
                #print 'changing learningRate to %f' % (learning_rate)
                #print 'failed to improve'
                #print 'saving'
            else:    
                pastCorrect=correct
                print 'saving best model'     
                bestModel=deepcopy(divas)
            
            totalcorrect+=correct
                    
        
        #lastModel=deepcopy(divas)
        print 'full validation %d of 10000' % (totalcorrect)
        end_epoch_time = timeit.default_timer()
        totalEpochTime = (end_epoch_time - epoch_time)
        print (('Epoch time: %f') % (totalEpochTime))
    print '...testing'
    correct=0            
    for s in range(0, 10000):
        outs = []
        outputs = []
        for c in range(0, 10):
            outs.append(proc(divas[c],dataset[2][0][s]))
            outputs.append(divas[c].visualize(dataset[2][0][s]))
        similarityCheck.append(numpy.array(outputs))

        pred=outs.index(min(outs))        
        act=int(dataset[2][1][s])
        if(pred==act):
            correct+=1
            """
            for x in range(0,0):
                vsl= numpy.reshape(fwdPass(fwdPass(dataset[2][0][s], divas[x].l1W),divas[x].l2W), (28,28))
                scipy.misc.imsave(('erre%dc%d.png'%(s,x)),vsl)
                
                vsl= numpy.reshape(dataset[2][0][s], (28,28))
                scipy.misc.imsave(('erre%dact.png'%(s)),vsl)
        else:
            for x in range(0,10):
                vsl= numpy.reshape(fwdPass(fwdPass(dataset[2][0][s], divas[x].l1W),divas[x].l2W), (28,28))
                scipy.misc.imsave(('erre%dc%d.png'%(s,x)),vsl)
                
                vsl= numpy.reshape(dataset[2][0][s], (28,28))
                scipy.misc.imsave(('erre%dact.png'%(s)),vsl)    
            break
        """
     #       print 'pred: %d actual: %d' % (pred, act)
    print 'final test %d of 10000' %(correct)
    print  '%f percent accuracy' % ((correct/10000.0)*100)
    end_time = timeit.default_timer()
    pastCorrect=correct
    print 'saving'     
    lastModel=deepcopy(divas)
    print 'done saving'
    training_time = (end_time - start_time)
    
    print (('Runtime: %f') % (training_time))
    similarityData=similarityCheck
    #similarityCheck [0-9999,10000-19999][0-10][0-783]
    channels = []
    for o in range(0, 10):
        pixels = []
        for z in range(0,783):
            pixel = []
            for r in range(0,9999):
                pixel.append(similarityCheck[r][o][z])
            pixels.append(numpy.array(pixel))
        channels.append(numpy.array(pixels))

    lastV = channels    

    for o in range(0, 10):
        for z in range(0, 783):
            for r in range(0,9999):
                channels[o][z][r]=1000*channels[o][z][r]
   
    avgs = []
    sd = []
    for o in range(0, 10):
        for z in range(0, 783):
            avgs.append(numpy.average(channels[o][z]))
            sd.append(numpy.std(channels[o][z]))
            """           
    diffs=[]
    for o in range(0, 10):
        for z in range(0, 783):
            k=numpy.average(channels[o][z])
            diffs.append(numpy.power((k-channels[o][z]),2))
            """  
    for o in range(0, 10):
        print("Average for channel %f: %f"%(o,(avgs[o])))  
        
        print("Average stdev for channel %f: %f"%(o,(sd[o]))) 
        
        #print("Squared error for channel %f: %f"%(o,diffs[o]) )

            
            
    #print >> sys.stderr, ('The file ' +
    #                      os.path.split(__file__)[1] +
    #                      ' ran for %.2fm' % ((training_time) / 60.))
    """                      
    divas[0].l1W.tofile("HidLayer")

    for x in range(0,10):
        divas[x].l2W.tofile(("layer%d"%(x)))
        
    """                        
      
    """           
    #scipy.misc.toimage(divas[0].l1W, cmin=0.0, cmax=1.0).save('hidden.png')
  
    
  #  for x in range(0,10):
   #     #scipy.misc.toimage(divas[x].l2W, cmin=0.0, cmax=1.0).save(('layer%d.png'%(x)))
    #    scipy.misc.imsave(('layer%d.png'%(x)), divas[x].l2W)
    
    vis = numpy.ones((784,), dtype=float) 
    print fwdPass(vis, divas[x].l1W).shape
    #vsl=numpy.reshape(fwdPass(vis, divas[x].l1W), (25,25)) 
    scipy.misc.imsave('hiddenLayer',vsl)       
    for x in range(0,10):
        vsl= numpy.reshape(fwdPass(fwdPass(vis, divas[x].l1W),divas[x].l2W), (28,28))
        scipy.misc.imsave(('vsl%d.png'%(x)),vsl)
        """
    
"""
end running MNIST section
start DIVA class section
"""    
    
    
    
class diva(object):
    def __init__(self, sharedWeights,sharedRNG):
        self.l1W=sharedWeights
        #self.l1b=sharedBiases
        self.rng=sharedRNG

        self.l2W=numpy.asarray(self.rng.uniform(
                    low=-numpy.sqrt(6. / (501+784)),
                    high=numpy.sqrt(6. / (501+784)),
                    size=(784, 501)
                ),
                dtype=numpy.float32)      
        
    def process(self,inputs):
        inVec=inputs
        hidVec=fwdPass(inVec,self.l1W)
        outVec=fwdPass(hidVec,self.l2W)
        totalSquaredError=numpy.sum(numpy.square(numpy.subtract(inVec,outVec))/2)
        return totalSquaredError
       
    def visualize(self,inputs):
        inVec=inputs
        hidVec=fwdPass(inVec,self.l1W)
        outVec=fwdPass(hidVec,self.l2W)       
        return outVec
       
    def train(self,inputs, learning_rate):    
        inVec=inputs
        hidVec=fwdPass(inVec,self.l1W)
        #print hidVec.shape
        outVec=fwdPass(hidVec,self.l2W)
        
        L2Werr=calcExampleError(hidVec,outVec,inVec)
 
        updateWeights(self.l2W, L2Werr, learning_rate)
        #L1Werr=hidVec*numpy.delete(L2Werr,0,axis=1)
        l1pd=numpy.dot((numpy.append(numpy.ones(1, dtype=numpy.float32), inVec)),numpy.transpose(self.l1W))
        L1Werr=numpy.dot(numpy.transpose(self.l2W),L2Werr)*numpy.append(numpy.ones(1, dtype=numpy.float32), (l1pd*(1-l1pd)))
        #print L1Werr.shape
        L1WerrSlice=L1Werr[1:501,1:501]
        L1WerrMat=backPass(inVec, numpy.dot(L1WerrSlice, numpy.transpose(hidVec)))
        #print L1WerrMat.shape
        #print self.l1W.shape
        updateWeights(self.l1W, L1WerrMat, learning_rate)
        #return numpy.subtract(inVec,outVec)
        totalSquaredError=numpy.sum(numpy.abs(numpy.subtract(inVec,outVec)))
        return totalSquaredError        
        
    def train2(self,inputs, learning_rate):    
        inVec=inputs
        hidVec=fwdPass(inVec,self.l1W)
        #print hidVec.shape
        outVec=fwdPass(hidVec,self.l2W)
        
        L2Werr=calcExampleError(hidVec,outVec,inVec)
 
        updateWeights(self.l2W, L2Werr, learning_rate)
        #L1Werr=hidVec*numpy.delete(L2Werr,0,axis=1)
        l1pd=numpy.dot((numpy.append(numpy.ones(1, dtype=numpy.float32), inVec)),numpy.transpose(self.l1W))
        L1Werr=numpy.dot(numpy.transpose(self.l2W),L2Werr)*numpy.append(numpy.ones(1, dtype=numpy.float32), (l1pd*(1-l1pd)))
        #print L1Werr.shape
        L1WerrSlice=L1Werr[1:501,1:501]
        L1WerrMat=backPass(inVec, numpy.dot(L1WerrSlice, numpy.transpose(hidVec)))
        #print L1WerrMat.shape
        #print self.l1W.shape
        #updateWeights(self.l1W, L1WerrMat, learning_rate)
        #return numpy.subtract(inVec,outVec)
        totalSquaredError=numpy.sum(numpy.abs(numpy.subtract(inVec,outVec)))
        return totalSquaredError  
        
"""
End DIVA class object section
Begin other functions section
"""
        
def proc(diva, inputs):
    inVec=inputs
    hidVec=fwdPass(inVec,diva.l1W)
    outVec=fwdPass(hidVec,diva.l2W)
    totalSquaredError=numpy.mean(numpy.square(numpy.subtract(inVec,outVec))/2)
    return totalSquaredError
    
def procc(diva, inputs):
    inVec=inputs
    hidVec=fwdPass(inVec,diva.l1W)
    outVec=fwdPass(hidVec,diva.l2W)
    return outVec    
    
def fwdPass(inVec, weights):
    biasedVec=(numpy.append(numpy.ones(1, dtype=numpy.float32), inVec))
    #print biasedVec            
    matProd=numpy.dot(biasedVec, numpy.transpose(weights))
    #print matProd
    retval = logSig(matProd)
    #print retval.dtype
    #print retval
    return retval 
    
def fwdPassl(inVec, weights):
    biasedVec=(numpy.append(numpy.ones(1, dtype=numpy.float32), inVec))
    #print biasedVec            
    matProd=numpy.dot(biasedVec, numpy.transpose(weights))
    #print matProd
    retval =(matProd)
    #print retval.dtype
    #print retval
    return retval 
            
def logSig(inVec):
    #print numpy.exp(-inVec)
    #print (1.0 + numpy.exp(-inVec))
    #print numpy.add(numpy.exp(-inVec),1)     
    return (1.0 / numpy.add(numpy.exp(-inVec),1)) 

def backPass(inVec, actual):
    biasedVec=(numpy.append(numpy.ones(1, dtype=numpy.float32), inVec))
    return (numpy.atleast_2d(biasedVec))*numpy.transpose(numpy.atleast_2d(actual))       
       
def calcElemErr(actual, expected):
    #print (actual-expected) * (actual) * (1-actual)
    return (actual-expected) * (actual) * (1-actual)

def calcExampleError(inVec, actual, expVec):
    biasedVec=(numpy.append(numpy.ones(1, dtype=numpy.float32), inVec))
    #print biasedVec.shape
    #once the vectors are cast as 2D arrays of length 1x3 and (1x2)^T  numpy can broadcast both to 2x3 to make operations possible- in this case, elementwise addition 
    #print numpy.atleast_2d(biasedVec)*numpy.transpose(numpy.atleast_2d(calcElemErr(actual, expVec))) 
    errs=numpy.transpose(numpy.atleast_2d(calcElemErr(actual, expVec))) 
    #print errs.shape
    return numpy.atleast_2d(biasedVec)*errs
            
def updateWeights(weights, errMat, learningRate):
    weights-=(errMat*learningRate)
    return weights
    
def chkChannels(divas):
    avgs=[]    
    sd=[]
    avgbias=[]
    for x in range(0,10):
        avgs.append(numpy.average(divas[x].l2W))
        sd.append(numpy.std(divas[x].l2W))
        bias = []        
        for k in range(0, 783):
            bias.append(divas[x].l2W[k][0])
        avgbias.append(numpy.average(bias))
        
    for o in range(0, 10):
        print("Average for channel %f: %f"%(o,(avgs[o])))  
        
        print("Average stdev for channel %f: %f"%(o,(sd[o]))) 
        
        print("Average bias for channel %f: %f"%(o,(avgbias[o]))) 


"""
end other functions section

start data loading section
Adapted from MILA lab code available at:
    https://github.com/lisa-lab/DeepLearningTutorials
"""
                

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split('__file__')[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '...loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    #print type(train_set)
    #print len(train_set)
    return numpy.asarray([train_set,valid_set,test_set])

"""
end data loading section
"""

#if __name__ == '__main__':
 #   test()
           