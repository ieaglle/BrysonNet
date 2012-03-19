/*
 * (c) Oleksandr Babii 2012
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using System.IO;
using System.Globalization;
using BrysonNet.ActivationFunctions;

namespace BrysonNet
{
    public class FeedForwardNeuralNetwork
    {
        #region Private fields

        private double _bias;
        private Random _random;
        private int _epoch;
        private IActivationFunction _activation;

        #region Neuron and layer counts

        private int _inputNeuronCount;
        private int _outputNeuronCount;
        private int[] _hiddenNeuronCount;
        private int _hiddenLayerCount;
        private int[] _neurons;

        #endregion

        private double[][] _signal;
        private double[][] _error;

        private double[][][] _weightJagged;
        private double[][][] _weightChangeJagged;
         
        #endregion

        #region Public properties

        /// <summary>
        /// Current input signals.
        /// </summary>
        public double[] InputSignal
        {
            get { return _signal[0]; }
            set { _signal[0] = value; }
        }

        /// <summary>
        /// Current output signals.
        /// </summary>
        public double[] OutputSignal
        {
            get { return _signal[_hiddenLayerCount + 1]; }
        }

        /// <summary>
        /// Current epoch.
        /// </summary>
        public int Epoch
        {
            get { return _epoch; }
            set { _epoch = value; }
        }

        /// <summary>
        /// Activation function.
        /// </summary>
        public IActivationFunction ActivationFunction
        {
            get { return _activation; }
            set { _activation = value; }
        }

        #endregion

        #region Constructors

        public FeedForwardNeuralNetwork() {}

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputNeuronCount">Number of input neurons.</param>
        /// <param name="hiddenNeuronCount">Number of hidden neurons.</param>
        /// <param name="outputNeuronCount">Number of output neurons.</param>
        public FeedForwardNeuralNetwork(int inputNeuronCount, int hiddenNeuronCount, int outputNeuronCount)
        {
            _inputNeuronCount = inputNeuronCount;
            _hiddenLayerCount = 1;
            _hiddenNeuronCount = new[] {hiddenNeuronCount};
            _outputNeuronCount = outputNeuronCount;
            _neurons = new[] {inputNeuronCount, hiddenNeuronCount, outputNeuronCount};
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputNeuronCount">Numbert of input neurons.</param>
        /// <param name="hiddenNeuronCount1">Number of hidden neurons of a first hidden layer.</param>
        /// <param name="hiddenNeuronCount2">Number of hidden neurons of a second hidden layer.</param>
        /// <param name="outputNeuronCount">Number of output neurons.</param>
        public FeedForwardNeuralNetwork(int inputNeuronCount, int hiddenNeuronCount1, int hiddenNeuronCount2, int outputNeuronCount)
            : this(inputNeuronCount, hiddenNeuronCount1, outputNeuronCount)
        {
            _hiddenLayerCount = 2;
            _hiddenNeuronCount = new[] {hiddenNeuronCount1, hiddenNeuronCount2};
            _neurons = new[] {inputNeuronCount, hiddenNeuronCount1, hiddenNeuronCount2, outputNeuronCount};
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputNeuronCount">Numbert of input neurons.</param>
        /// <param name="hiddenNeuronCount1">Number of hidden neurons of a first hidden layer.</param>
        /// <param name="hiddenNeuronCount2">Number of hidden neurons of a second hidden layer.</param>
        /// <param name="hiddenNeuronCount3">Number of hidden neurons of a third hidden layer.</param>
        /// <param name="outputNeuronCount">Number of output neurons.</param>
        public FeedForwardNeuralNetwork(int inputNeuronCount, int hiddenNeuronCount1, int hiddenNeuronCount2, int hiddenNeuronCount3, int outputNeuronCount)
            : this(inputNeuronCount, hiddenNeuronCount1, hiddenNeuronCount2, outputNeuronCount)
        {
            _hiddenLayerCount = 3;
            _hiddenNeuronCount = new[] {hiddenNeuronCount1, hiddenNeuronCount2, hiddenNeuronCount3};
            _neurons = new[]
                {_inputNeuronCount, hiddenNeuronCount1, hiddenNeuronCount2, hiddenNeuronCount3, outputNeuronCount};
        }

        #endregion

        /// <summary>
        /// Initializes averything.
        /// </summary>
        public void Initialize()
        {
            _random = new Random();
            _epoch = 0;
            _activation = new Sigmoid();

            #region Jagged weight

            _weightJagged = new double[_hiddenLayerCount + 1][][];
            _weightChangeJagged = new double[_hiddenLayerCount + 1][][];

            for (int i = 0; i < _hiddenLayerCount + 1; i++)
            {
                _weightJagged[i] = new double[_neurons[i]][];
                _weightChangeJagged[i] = new double[_neurons[i]][];
                for (int j = 0; j < _neurons[i]; j++)
                {
                    _weightJagged[i][j] = new double[_neurons[i + 1]];
                    _weightChangeJagged[i][j] = new double[_neurons[i + 1]];
                }
            }

            #endregion

            #region Signals and errors initialization

            _signal = new double[_hiddenLayerCount + 2][];
            _error = new double[_hiddenLayerCount + 2][];

            //amount of layers = input + output + all hidden
            _signal[0] = new double[_inputNeuronCount];
            _error[0] = new double[_inputNeuronCount];

            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                _signal[i + 1] = new double[_hiddenNeuronCount[i]];
                _error[i + 1] = new double[_hiddenNeuronCount[i]];
            }

            _signal[_hiddenLayerCount + 1] = new double[_outputNeuronCount];
            _error[_hiddenLayerCount + 1] = new double[_outputNeuronCount];

            #endregion
        }

        /// <summary>
        /// Pulsing the network.
        /// </summary>
        public void Pulse()
        {
            //for all layers
            for (int layer = 1; layer < _hiddenLayerCount + 2; layer++) //plus input and output layers
            {
                //for each neuron from current layer
                for (int i = 0; i < _neurons[layer]; i++)//_signal[layer].GetLength(0); i++)
                {
                    double temp = 0;
                    //for each neuron from previous layer
                    for (int j = 0; j < _neurons[layer-1]; j++)//_signal[layer - 1].GetLength(0); j++)
                    {
                        temp += _signal[layer - 1][j] * _weightJagged[layer - 1][j][i];
                    }
                    //only for second layer add additional bias
                    _signal[layer][i] = layer == 1 ? _activation.Calc(temp + _bias) : _activation.Calc(temp);
                }
            }
            _epoch++;
        }

        /// <summary>
        /// Training the network with specified inputs and desired outputs.
        /// </summary>
        /// <param name="input">Input values.</param>
        /// <param name="desiredOutput">Desired output values.</param>
        /// <param name="precision">How precise network should be (if equal 0.1 network will stop when |desired output - actual output| is less than 0.1 ).</param>
        /// <param name="learningRate">How fast network is learning.</param>
        public void Train(double[][] input, double[][] desiredOutput, double precision, double learningRate = 1)
        {
            if (input[0].Count() != _inputNeuronCount || desiredOutput[0].Count() != _outputNeuronCount)
                throw new Exception("Input or output values number are invalid.");

            // maximum epoches allowed
            const int maxEpoches = 5000000;
            bool go = true;
            int sequences = input.Count();
            while (go)
            {
                // check for infinite loop
                if (_epoch >= maxEpoches)
                    throw new Exception("Training takes too long. Try to change network architecture.");
                //results[input seq][results]
                double[][] results = new double[sequences][];
                for (int curr = 0; curr < sequences; curr++)
                {
                    double[] currInput = input[curr];
                    double[] currDesiredOutput = desiredOutput[curr];
                    //setting input signals
                    for (int i = 0; i < _inputNeuronCount; i++)
                    {
                        _signal[0][i] = currInput[i];
                    }
                    //executing network
                    Pulse();
                    //reading results
                    results[curr] = new double[_outputNeuronCount];
                    //calculating errors of output neurons 
                    for (int i = 0; i < _outputNeuronCount; i++)
                    {
                        _error[_hiddenLayerCount + 1][i] =
                            _activation.Derivative(_signal[_hiddenLayerCount + 1][i])*
                            (currDesiredOutput[i] - _signal[_hiddenLayerCount + 1][i]);
                        results[curr][i] = _signal[_hiddenLayerCount + 1][i];
                    }
                    //calculating all errors and wight changes
                    for (int layer = _hiddenLayerCount + 1; layer > 0; layer--)
                    {
                        for (int i = 0; i < _neurons[layer-1]; i++)
                        {
                            double temp = 0;
                            for (int j = 0; j < _neurons[layer]; j++)
                            {
                                double delta = learningRate * _error[layer][j] * _signal[layer - 1][i];
                                //adding previous weight change multiplied by a constant 
                                //to prevent from detecting a local minima
                                _weightJagged[layer - 1][i][j] += delta + 0.5 * _weightChangeJagged[layer - 1][i][j];
                                _weightChangeJagged[layer - 1][i][j] = delta;
                                temp += _error[layer][j] * _weightJagged[layer - 1][i][j];
                            }
                            _error[layer - 1][i] = _activation.Derivative(_signal[layer - 1][i]) * temp;
                        }
                    }
                }
                //check if network is trained enough
                for (int i = 0; i < sequences; i++)
                {
                    for (int j = 0; j < _outputNeuronCount; j++)
                    {
                        go = go && Math.Abs(desiredOutput[i][j] - results[i][j]) >= precision;
                    }
                }
            }
        }

        /*
        
        /// <summary>
        /// Pulsing the network.
        /// </summary>
        private void ParallelPulse(ref double[][] signal, ref double[][][] weight)
        {
            //for all layers
            for (int layer = 1; layer < _hiddenLayerCount + 2; layer++) //plus input and output layers
            {
                //for each neuron from current layer
                for (int i = 0; i < _neurons[layer]; i++)//_signal[layer].GetLength(0); i++)
                {
                    double temp = 0;
                    //for each neuron from previous layer
                    for (int j = 0; j < _neurons[layer - 1]; j++)//_signal[layer - 1].GetLength(0); j++)
                    {
                        temp += signal[layer - 1][j] * weight[layer - 1][j][i];
                    }
                    //only for second layer add additional bias
                    _signal[layer][i] = layer == 1 ? _activation.Calc(temp + _bias) : _activation.Calc(temp);
                }
            }
            _epoch++;
        } 
         
        /// <summary>
        /// Parallel training the network with specified inputs and desired outputs.
        /// </summary>
        /// <param name="input">Input values.</param>
        /// <param name="desiredOutput">Desired output values.</param>
        /// <param name="precision">How precise network should be (if equal 0.1 network will stop when |desired output - actual output| is less than 0.1 ).</param>
        /// <param name="learningRate">How fast network is learning.</param>
        public void ParallelTrain(double[][] input, double[][] desiredOutput, double precision, double learningRate = 1)
        {
            if (input[0].Count() != _inputNeuronCount || desiredOutput[0].Count() != _outputNeuronCount)
                throw new Exception("Input or output values number are invalid.");

            // maximum epoches allowed
            const int maxEpoches = 5000000;
            bool go = true;
            int sequences = input.Count();
            while (_epoch < 100000)
            {
                // check for infinite loop
                if (_epoch >= maxEpoches)
                    throw new Exception("Training takes too long. Try to change network architecture.");
                //results[input seq][results]
                double[][] results = new double[sequences][];
                //for (int curr = 0; curr < sequences; curr++)

                double[][][] localSignals = new double[sequences][][];
                double[][][] localErrors = new double[sequences][][];
                double[][][][] localWeights = new double[sequences][][][];
                double[][][][] localWeightsChanges = new double[sequences][][][];

                Parallel.For(0, sequences, curr =>
                    {
                        double[][] localSignal = (double[][]) _signal.Clone();
                        double[][] localError = (double[][]) _error.Clone();

                        double[][][] localWeight = (double[][][]) _weightJagged.Clone();
                        double[][][] localWeightChange = (double[][][]) _weightChangeJagged.Clone();

                        localSignals[curr] = (double[][])_signal.Clone();
                        localErrors[curr] = (double[][])_error.Clone();

                        localWeights[curr] = (double[][][])_weightJagged.Clone();
                        localWeightsChanges[curr] = (double[][][])_weightChangeJagged.Clone();

                        double[] currInput = input[curr];
                        double[] currDesiredOutput = desiredOutput[curr];
                        //setting input signals
                        for (int i = 0; i < _inputNeuronCount; i++)
                        {
                            localSignals[curr][0][i] = currInput[i];
                        }
                        //executing network
                        ParallelPulse(ref localSignals[curr], ref localWeights[curr]);
                        //reading results
                        results[curr] = new double[_outputNeuronCount];
                        //calculating errors of output neurons 
                        for (int i = 0; i < _outputNeuronCount; i++)
                        {
                            localErrors[curr][_hiddenLayerCount + 1][i] =
                                _activation.Derivative(localSignals[curr][_hiddenLayerCount + 1][i])*
                                (currDesiredOutput[i] - localSignals[curr][_hiddenLayerCount + 1][i]);
                            results[curr][i] = localSignals[curr][_hiddenLayerCount + 1][i];
                        }
                        //calculating all errors and wight changes
                        for (int layer = _hiddenLayerCount + 1; layer > 0; layer--)
                        {
                            for (int i = 0; i < _neurons[layer - 1]; i++)
                            {
                                double temp = 0;
                                for (int j = 0; j < _neurons[layer]; j++)
                                {
                                    double delta = learningRate*localErrors[curr][layer][j]*localSignals[curr][layer - 1][i];
                                    //adding previous weight change multiplied by a constant 
                                    //to prevent from detecting a local minima
                                    localWeights[curr][layer - 1][i][j] += delta + 0.5*_weightChangeJagged[layer - 1][i][j];
                                    localWeightsChanges[curr][layer - 1][i][j] = delta;
                                    temp += _error[layer][j]*localWeights[curr][layer - 1][i][j];
                                }
                                localErrors[curr][layer - 1][i] = _activation.Derivative(localSignals[curr][layer - 1][i])*temp;
                            }
                        }
                    });


                for (int layer = 0; layer < _hiddenLayerCount + 1; layer++)
                {
                    for (int i = 0; i < _neurons[layer]; i++)
                    {
                        for (int j = 0; j < _neurons[layer + 1]; j++)
                        {
                            for (int seq = 0; seq < sequences; seq++)
                            {
                                _weightJagged[layer][i][j] += localWeightsChanges[seq][layer][i][j];
                                _weightChangeJagged[layer][i][j] += localWeightsChanges[seq][layer][i][j];
                            }
                        }
                    }
                }

                //check if network is trained enough
                for (int i = 0; i < sequences; i++)
                {
                    for (int j = 0; j < _outputNeuronCount; j++)
                    {
                        go = go && Math.Abs(desiredOutput[i][j] - results[i][j]) >= precision;
                    }
                }
            }
        }
*/

        /// <summary>
        /// Randomizing all weights.
        /// </summary>
        /// <param name="minValue">Minimum weight value.</param>
        /// <param name="maxValue">Maximum weight value.</param>
        public void RandomizeWeights(double minValue = 0, double maxValue = 1)
        {
            if (maxValue <= minValue)
                throw new Exception("Maximum has to be greater then minimum, didn't you know?");

            _bias = _random.NextDouble();

            for (int layer = 0; layer < _hiddenLayerCount + 1; layer++)
            {
                for (int i = 0; i < _neurons[layer]; i++)
                {
                    for (int j = 0; j < _neurons[layer+1]; j++)
                    {
                        _weightJagged[layer][i][j] = _random.NextDouble() * (maxValue - minValue) + minValue;
                    }
                }
            }
        }

        /// <summary>
        /// Saving current network's state.
        /// </summary>
        /// <param name="filename">Name of the file for storing network's state. Will be "filename".xml.</param>
        public void Save(string filename)
        {
            XDocument doc = new XDocument(new XElement("root"));
            doc.Root.Add(
                new XAttribute("inputneurons", _inputNeuronCount),
                new XAttribute("hiddenlayers", _hiddenLayerCount));

            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                doc.Root.Add(new XAttribute("hidden"+i, _hiddenNeuronCount[i]));
            }
            doc.Root.Add(new XAttribute("outputneurons", _outputNeuronCount));

            XElement inputWeights = new XElement("weights");
            for (int layer = 0; layer < _hiddenLayerCount+1; layer++)
            {
                for (int i = 0; i < _neurons[layer]; i++)//_weight[layer].GetLength(0); i++)
                {
                    for (int j = 0; j < _neurons[layer+1]; j++)//_weight[layer].GetLength(1); j++)
                    {
                        XElement element = new XElement("weight");
                        element.Add(
                            new XAttribute("layer", layer),
                            new XAttribute("from", i),
                            new XAttribute("to", j),
                            new XAttribute("value", _weightJagged[layer][i][j]),
                            new XAttribute("weightchange", _weightChangeJagged[layer][i][j]));
                        inputWeights.Add(element);
                    }
                }
            }
            doc.Root.Add(
                new XElement(inputWeights));
            doc.Save(filename);
        }
        
        /// <summary>
        /// Loading network from a specified file.
        /// </summary>
        /// <param name="filename">Name of the file.</param>
        public void Load(string filename)
        {
            if (String.IsNullOrWhiteSpace(filename) || !File.Exists(filename))
                throw new Exception("File path is invalid");

            using (FileStream stream = new FileStream(filename, FileMode.Open))
            {
                try
                {
                    XDocument doc = XDocument.Load(stream);
                    int inputNeuronCount = int.Parse(doc.Root.Attribute("inputneurons").Value);
                    int hiddenLayerCount = int.Parse(doc.Root.Attribute("hiddenlayers").Value);
                    int[] hiddenNeuronCount = new int[hiddenLayerCount];
                    for (int i = 0; i < hiddenLayerCount; i++)
                    {
                        hiddenNeuronCount[i] = int.Parse(doc.Root.Attribute("hidden" + i).Value);
                    }
                    int outputNeuronCount = int.Parse(doc.Root.Attribute("outputneurons").Value);

                    _inputNeuronCount = inputNeuronCount;
                    _hiddenLayerCount = hiddenLayerCount;
                    _hiddenNeuronCount = hiddenNeuronCount;
                    _outputNeuronCount = outputNeuronCount;
                    _neurons = new int[hiddenLayerCount + 2];
                    _neurons[0] = inputNeuronCount;
                    for (int i = 0; i < hiddenLayerCount; i++)
                    {
                        _neurons[i + 1] = hiddenNeuronCount[i];
                    }
                    _neurons[hiddenLayerCount + 1] = outputNeuronCount;

                    Initialize();
                    IEnumerable<XElement> weights = doc.Element("root").Elements("weights");
                    foreach (var xElement in weights.Elements("weight"))
                    {
                        int layer = int.Parse(xElement.Attribute("layer").Value);
                        int from = int.Parse(xElement.Attribute("from").Value);
                        int to = int.Parse(xElement.Attribute("to").Value);
                        double weight = double.Parse(xElement.Attribute("value").Value, CultureInfo.InvariantCulture);
                        double weightChange = double.Parse(xElement.Attribute("weightchange").Value,
                                                           CultureInfo.InvariantCulture);
                        _weightJagged[layer][from][to] = weight;
                        // probably not necessary, unless future training will be needed
                        _weightChangeJagged[layer][from][to] = weightChange;
                    }
                }
                catch (Exception)
                {
                    throw new Exception("Project file is corrupted.");
                }
            }
        }
    }
}
