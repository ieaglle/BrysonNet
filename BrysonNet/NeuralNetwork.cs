using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Xml.Linq;
using BrysonNet.ActivationFunctions;
using BrysonNet.Training;

namespace BrysonNet
{
    public abstract class NeuralNetwork
    {
        #region Protected fields

        protected double _bias;
        protected Random _random;

        protected int _epoch;
        protected IActivationFunction _activation;
        
        protected int _inputNeuronCount;
        protected int _outputNeuronCount;
        protected int[] _hiddenNeuronCount;
        protected int _hiddenLayerCount;
        protected int[] _neurons;

        protected double[][] _signal;
        protected double[][] _error;

        protected double[][][] _weight;
        protected double[][][] _weightChange;

        #endregion

        #region Public properties

        /// <summary>
        /// Number of layers in the network.
        /// </summary>
        public int Layers
        {
            get { return _hiddenLayerCount + 2; }
        }

        /// <summary>
        /// Number of neurons in each layer.
        /// </summary>
        public int[] Neurons
        {
            get { return _neurons; }
        }

        /// <summary>
        /// Current input signal.
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
        }

        /// <summary>
        /// Activation function.
        /// </summary>
        public IActivationFunction ActivationFunction
        {
            get { return _activation; }
            set { _activation = value; }
        }

        /// <summary>
        /// Current errors.
        /// </summary>
        public double[][] Error
        {
            get { return _error; }
            set { _error = value; }
        }

        /// <summary>
        /// Signals of all neurons in each layer.
        /// </summary>
        public double[][] Signal
        {
            get { return _signal; }
        }

        /// <summary>
        /// Current weights.
        /// </summary>
        public double[][][] Weight
        {
            get { return _weight; }
            set { _weight = value; }
        }

        public double[][][] WeightChange
        {
            get { return _weightChange; }
            set { _weightChange = value; }
        }

        #endregion

        /// <summary>
        /// Initializes averything.
        /// </summary>
        public virtual void Initialize()
        {
            _random = new Random();
            _epoch = 0;
            if (_activation == null)
                _activation = new Sigmoid();

            #region weights

            _weight = new double[_hiddenLayerCount + 1][][];
            _weightChange = new double[_hiddenLayerCount + 1][][];

            for (int i = 0; i < _hiddenLayerCount + 1; i++)
            {
                _weight[i] = new double[_neurons[i]][];
                _weightChange[i] = new double[_neurons[i]][];
                for (int j = 0; j < _neurons[i]; j++)
                {
                    _weight[i][j] = new double[_neurons[i + 1]];
                    _weightChange[i][j] = new double[_neurons[i + 1]];
                }
            }

            #endregion

            #region signals and errors

            _signal = new double[_hiddenLayerCount + 2][];
            _error = new double[_hiddenLayerCount + 2][];

            for (int i = 0; i < _hiddenLayerCount + 2; i++)
            {
                _signal[i] = new double[_neurons[i]];
                _error[i] = new double[_neurons[i]];
            }

            #endregion
        }

        /// <summary>
        /// Pulsing the network.
        /// </summary>
        public virtual void Pulse()
        {
            //for all layers
            for (int layer = 1; layer < _hiddenLayerCount + 2; layer++) //plus input and output layers
            {
                //for each neuron from current layer
                for (int i = 0; i < _neurons[layer]; i++) //_signal[layer].GetLength(0); i++)
                {
                    double temp = 0;
                    //for each neuron from previous layer
                    for (int j = 0; j < _neurons[layer - 1]; j++) //_signal[layer - 1].GetLength(0); j++)
                    {
                        temp += _signal[layer - 1][j] * _weight[layer - 1][j][i];
                    }
                    //only for second layer add additional bias
                    _signal[layer][i] = layer == 1 ? _activation.Calc(temp + _bias) : _activation.Calc(temp);
                }
            }
            _epoch++;
        }

        /// <summary>
        /// Randomizing all weights.
        /// </summary>
        /// <param name="minValue">Minimum weight value.</param>
        /// <param name="maxValue">Maximum weight value.</param>
        public virtual void RandomizeWeights(double minValue = 0, double maxValue = 1)
        {
            if (maxValue <= minValue)
                throw new Exception("Maximum has to be greater then minimum, didn't you know?");

            _bias = _random.NextDouble();

            for (int layer = 0; layer < _hiddenLayerCount + 1; layer++)
            {
                for (int i = 0; i < _neurons[layer]; i++)
                {
                    for (int j = 0; j < _neurons[layer + 1]; j++)
                    {
                        _weight[layer][i][j] = _random.NextDouble() * (maxValue - minValue) + minValue;
                    }
                }
            }
        }

        /// <summary>
        /// Training the network with specified inputs and desired outputs.
        /// </summary>
        /// <param name="input">Input values.</param>
        /// <param name="desiredOutput">Desired output values.</param>
        public abstract void Train(double[][] input, double[][] desiredOutput);

        #region Saving/Loading network

        /// <summary>
        /// Saving current network's state.
        /// </summary>
        /// <param name="filename">Name of the file for storing network's state. Will be "filename".xml.</param>
        public virtual void Save(string filename)
        {
            XDocument doc = new XDocument(new XElement("root"));
            doc.Root.Add(
                new XAttribute("inputneurons", _inputNeuronCount),
                new XAttribute("hiddenlayers", _hiddenLayerCount));

            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                doc.Root.Add(new XAttribute("hidden" + i, _hiddenNeuronCount[i]));
            }
            doc.Root.Add(new XAttribute("outputneurons", _outputNeuronCount));

            XElement inputWeights = new XElement("weights");
            for (int layer = 0; layer < _hiddenLayerCount + 1; layer++)
            {
                for (int i = 0; i < _neurons[layer]; i++)//_weight[layer].GetLength(0); i++)
                {
                    for (int j = 0; j < _neurons[layer + 1]; j++)//_weight[layer].GetLength(1); j++)
                    {
                        XElement element = new XElement("weight");
                        element.Add(
                            new XAttribute("layer", layer),
                            new XAttribute("from", i),
                            new XAttribute("to", j),
                            new XAttribute("value", _weight[layer][i][j]),
                            new XAttribute("weightchange", _weightChange[layer][i][j]));
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
        public virtual void Load(string filename)
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
                        _weight[layer][from][to] = weight;
                        // probably not necessary, unless future training will be needed
                        _weightChange[layer][from][to] = weightChange;
                    }
                }
                catch (Exception)
                {
                    throw new Exception("Project file is corrupted.");
                }
            }
        }

        #endregion
    }
}
