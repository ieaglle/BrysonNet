/*
 * (c) Oleksandr Babii 2012
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using System.IO;
using System.Globalization;

namespace BrysonNet
{
    public class NeuralNetwork : INeuralNetwork
    {
        #region Private fields

        private double _bias;
        private Random _random;
        private int _epoch;

        #region Neuron and layer counts

        private int _inputNeuronCount;
        private int _outputNeuronCount;
        private int[] _hiddenNeuronCount;
        private int _hiddenLayerCount;

        #endregion

        private Dictionary<int, double[,]> _weight;
        private Dictionary<int, double[,]> _weightChange;
        private Dictionary<int, double[]> _signal;
        private Dictionary<int, double[]> _error;
         
        #endregion

        #region Public properties

        public double[] InputSignal
        {
            get { return _signal[0]; }
            set { _signal[0] = value; }
        }

        public double[] OutputSignal
        {
            get { return _signal[_hiddenLayerCount + 1]; }
            set { _signal[_hiddenLayerCount + 2] = value; }
        }

        public int Epoch
        {
            get { return _epoch; }
            set { _epoch = value; }
        }

        #endregion

        #region Constructors

        public NeuralNetwork()
        {
            
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputNeuronCount">Number of input neurons.</param>
        /// <param name="hiddenNeuronCount">Number of hidden neurons.</param>
        /// <param name="outputNeuronCount">Number of output neurons.</param>
        public NeuralNetwork(int inputNeuronCount, int hiddenNeuronCount, int outputNeuronCount)
        {
            _inputNeuronCount = inputNeuronCount;
            _hiddenLayerCount = 1;
            _hiddenNeuronCount = new[] {hiddenNeuronCount};
            _outputNeuronCount = outputNeuronCount;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputNeuronCount">Numbert of input neurons.</param>
        /// <param name="hiddenNeuronCount1">Number of hidden neurons of a first hidden layer.</param>
        /// <param name="hiddenNeuronCount2">Number of hidden neurons of a second hidden layer.</param>
        /// <param name="outputNeuronCount">Number of output neurons.</param>
        public NeuralNetwork(int inputNeuronCount, int hiddenNeuronCount1, int hiddenNeuronCount2, int outputNeuronCount)
            : this(inputNeuronCount, hiddenNeuronCount1, outputNeuronCount)
        {
            _hiddenLayerCount = 2;
            _hiddenNeuronCount = new[] {hiddenNeuronCount1, hiddenNeuronCount2};
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputNeuronCount">Numbert of input neurons.</param>
        /// <param name="hiddenNeuronCount1">Number of hidden neurons of a first hidden layer.</param>
        /// <param name="hiddenNeuronCount2">Number of hidden neurons of a second hidden layer.</param>
        /// <param name="hiddenNeuronCount3">Number of hidden neurons of a third hidden layer.</param>
        /// <param name="outputNeuronCount">Number of output neurons.</param>
        public NeuralNetwork(int inputNeuronCount, int hiddenNeuronCount1, int hiddenNeuronCount2, int hiddenNeuronCount3, int outputNeuronCount)
            : this(inputNeuronCount, hiddenNeuronCount1, hiddenNeuronCount2, outputNeuronCount)
        {
            _hiddenLayerCount = 3;
            _hiddenNeuronCount = new[] {hiddenNeuronCount1, hiddenNeuronCount2, hiddenNeuronCount3};
        }

        #endregion

        /// <summary>
        /// Initializes averything.
        /// </summary>
        public void Initialize()
        {
            _random = new Random();
            _epoch = 0;

            #region Weights initialization

            _weight = new Dictionary<int, double[,]>(1 + _hiddenLayerCount);
            _weightChange = new Dictionary<int, double[,]>(1 + _hiddenLayerCount);
            _weight.Add(0, new double[_inputNeuronCount,_hiddenNeuronCount[0]]);
            _weightChange.Add(0, new double[_inputNeuronCount,_hiddenNeuronCount[0]]);
            for (int i = 0; i < _hiddenLayerCount-1; i++)
            {
                _weight.Add(i + 1, new double[_hiddenNeuronCount[i],_hiddenNeuronCount[i + 1]]);
                _weightChange.Add(i + 1, new double[_hiddenNeuronCount[i],_hiddenNeuronCount[i + 1]]);
            }
            _weight.Add(_weight.Count, new double[_hiddenNeuronCount[_hiddenLayerCount - 1],_outputNeuronCount]);
            _weightChange.Add(_weightChange.Count,
                              new double[_hiddenNeuronCount[_hiddenLayerCount - 1],_outputNeuronCount]);
            #endregion

            #region Signals and errors initialization

            //amount of layers = input + output + all hidden
            _signal = new Dictionary<int, double[]>(2 + _hiddenLayerCount);
            _error = new Dictionary<int, double[]>(2 + _hiddenLayerCount);
            _signal.Add(0, new double[_inputNeuronCount]);
            _error.Add(0, new double[_inputNeuronCount]);
            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                _signal.Add(i + 1, new double[_hiddenNeuronCount[i]]);
                _error.Add(i + 1, new double[_hiddenNeuronCount[i]]);
            }
            _signal.Add(_signal.Count, new double[_outputNeuronCount]);
            _error.Add(_error.Count, new double[_outputNeuronCount]);
            #endregion
        }

        /// <summary>
        /// Pulsing the network.
        /// </summary>
        public void Pulse()
        {
            //RandomizeWeights();
            //for all layers
            for (int layer = 1; layer < _hiddenLayerCount + 2; layer++) //plus input and output layers
            {
                //for each neuron from current layer
                for (int i = 0; i < _signal[layer].GetLength(0); i++)
                {
                    double temp = 0;
                    //for each neuron from previous layer
                    for (int j = 0; j < _signal[layer-1].GetLength(0); j++)
                    {
                        temp += _signal[layer - 1][j]*_weight[layer - 1][j, i];
                    }
                    //for the first layer add additional bias
                    _signal[layer][i] = layer == 1 ? Sigmoid(temp + _bias) : Sigmoid(temp); 
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
                throw new Exception("Input or output values number is invalid.");

            // maximum epoches allowed
            const int maxEpoches = 500000;
            bool go = true;
            do
            {
                // check for infinite loop
                if (_epoch >= maxEpoches)
                    throw new Exception("Training takes too long. Try to change network architecture.");

                double[][] results = new double[input.Count()][];
                for (int curr = 0; curr < input.Count(); curr++)
                {
                    double[] currInput = input[curr];
                    double[] currDesiredOutput = desiredOutput[curr];
                    // setting input signals
                    //InputSignal = currInput;
                    for (int i = 0; i < _inputNeuronCount; i++)
                    {
                        _signal[0][i] = currInput[i];
                    }
                    // executing network
                    Pulse();
                    results[curr] = new double[_outputNeuronCount];
                    // calculating errors of output neurons
                    for (int i = 0; i < _outputNeuronCount; i++)
                    {
                        _error[_hiddenLayerCount + 1][i] = _signal[_hiddenLayerCount + 1][i]*
                                                           (1.0 - _signal[_hiddenLayerCount + 1][i])*
                                                           (currDesiredOutput[i] - _signal[_hiddenLayerCount + 1][i]);
                        results[curr][i] = _signal[_hiddenLayerCount + 1][i];
                    }
                    //calculating all errors and wight changes
                    for (int layer = _hiddenLayerCount + 1; layer > 0; layer--)
                    {
                        for (int i = 0; i < _signal[layer - 1].Count(); i++)
                        {
                            double temp = 0;
                            for (int j = 0; j < _signal[layer].Count(); j++)
                            {
                                double delta = learningRate*_error[layer][j]*_signal[layer - 1][i];
                                // adding previous weight change multiplied by a constant 
                                // to prevent from finding local minima
                                _weight[layer - 1][i, j] += delta + 0.5*_weightChange[layer - 1][i, j];
                                _weightChange[layer - 1][i, j] = delta;
                                temp += _error[layer][j]*_weight[layer - 1][i, j];
                            }
                            _error[layer - 1][i] = _signal[layer - 1][i]*(1.0 - _signal[layer - 1][i])*temp;
                        }
                    }
                }

                // check if network is trained enought
                for (int i = 0; i < input.Count(); i++)
                {
                    for (int j = 0; j < _outputNeuronCount; j++)
                    {
                        go = go && Math.Abs(desiredOutput[i][j] - results[i][j]) >= precision;
                    }
                }

            } while (go);
        }

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
                for (int i = 0; i < _weight[layer].GetLength(0); i++)
                {
                    for (int j = 0; j < _weight[layer].GetLength(1); j++)
                    {
                        _weight[layer][i, j] = _random.NextDouble() * (maxValue - minValue) + minValue;
                    }
                }
            }
        }

        public void Show()
        {
            for (int layer = 0; layer < _hiddenLayerCount + 2; layer++)
            {
                Console.WriteLine("\nLayer " + layer);

                Console.WriteLine("Signals:");
                for (int i = 0; i < _signal[layer].Count(); i++)
                {
                    Console.Write(_signal[layer][i] + " ");
                }
                Console.WriteLine();
            }
            for (int layer = 0; layer < _hiddenLayerCount + 1; layer++)
            {
                Console.Out.WriteLine("\nWeights:");
                for (int i = 0; i < _weight[layer].GetLength(0); i++)
                {
                    for (int j = 0; j < _weight[layer].GetLength(1); j++)
                    {
                        Console.Out.WriteLine(_weight[layer][i, j]);
                    }
                }
            }
        }

        private double Sigmoid(double net)
        {
            return 1/(1 + Math.Exp(-net));
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
                for (int i = 0; i < _weight[layer].GetLength(0); i++)
                {
                    for (int j = 0; j < _weight[layer].GetLength(1); j++)
                    {
                        XElement element = new XElement("weight");
                        element.Add(
                            new XAttribute("layer", layer),
                            new XAttribute("from", i),
                            new XAttribute("to", j),
                            new XAttribute("value", _weight[layer][i, j]),
                            new XAttribute("weightchange", _weightChange[layer][i,j]));
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
                        _weight[layer][from, to] = weight;
                        // probably not necessary, unless future training will be performed
                        _weightChange[layer][from, to] = weightChange;
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
