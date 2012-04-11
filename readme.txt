I've just started to learn neural networks and "BrysonNet" is my first attempt to create them using C#.

In the future I'd like to create some handwritten/typed text recognition and image processing using this project.

Project is under MS-PL license.

Current features:
 - it's possible to create neural network with custom numbers of neurons;
 - from one to three hidden layers available (in future will be more if necessary);
 - weights randomizing;
 - network pulsing;
 - backpropagation training;
 - XOR operation example is included;
 - saving/loading network's state to XML file;
 - small knowledge base is provided (for now only XOR :)).


TODO's:
 -FeedForward neural network:
   [ ] correct adaptive backpropagation training;
 -General: 
   [ ] implement Self Organizing Feature Map (SOFM);
   [ ] implement Hopfield Neural Network;
   [ ] implement Simple Recurrent Network (SRN) Elman Style;
   [ ] implement SRN Jordan Style;
   [ ] implement SRN Self Organizing Map;
   [ ] implement Feedforward Radial Basis Function (RBF) Network.


Releases history:
11-04-2012
 - partially implemented Hopfield network;
 - partially implemented SOFM;
 - first attempts of genetic algorithms.

22-03-2012
 - implemented adaptive backpropagation (in some cases network convergences much faster);
 - reorganised project structure;

19-03-2012
 - renamed "NeuralNetwork" with a proper name "FeedForwardNeuralNetwork";
 - optimized performance up to 450 times! using jagged arrays for weights instead of dictionaries and some code optimizations;
 - increased maximum epoches to 5000000 (10 times more);
 - added bipolar sigmoid function;
 - choosing activation functions is now more OOP-ish :).

27-02-2012
 - changed project output type to Class Library;
 - added new project BrysonNetUsage for features testing;
 - added protection from infinite loop while training (after 500000 iterations exception will be thrown);
 - added saving of current network's state in XML file;
 - added loading network's state from XML file;
 - added directory "kb" for storing pre-trained networks;
 - added pre-trained network for solving XOR operation. Heh.


(c) Oleksandr Babii, BrysonNet, 2012, Kyiv
