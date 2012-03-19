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
   [x] add weight ranges for randomizing (e.g. -1 to 1, or -0,5 to 0,5 etc.);
   [x] add error and previous changes arrays;
   [x] implement back propagation training (more in future);
   [x] add some protection from infinite loop while training;
   [x] add possibility to save current state in XML file;
   [x] add possibility to load network from a XML file;
 -General: 
   [x] move NeuralNetwork class to external assembly to make it as a dll, not a console application;
   [>] add multithreading;
   [ ] decide to use SQLite or similar;
   [ ] implement Self Organizing Feature Map (SOFM);
   [ ] implement Hopfield Neural Network;
   [ ] implement Simple Recurrent Network (SRN) Elman Style;
   [ ] implement SRN Jordan Style;
   [ ] implement SRN Self Organizing Map;
   [ ] implement Feedforward Radial Basis Function (RBF) Network.


Releases history:

vNext
 - renamed "NeuralNetwork" with a proper name "FeedForwardNeuralNetwork";

27-02-2012
 - changed project output type to Class Library;
 - added new project BrysonNetUsage for features testing;
 - added protection from infinite loop while training (after 500000 iterations exception will be thrown);
 - added saving of current network's state in XML file;
 - added loading network's state from XML file;
 - added directory "kb" for storing pre-trained networks;
 - added pre-trained network for solving XOR operation. Heh.


(c) Oleksandr Babii, BrysonNet, 2012, Kyiv
