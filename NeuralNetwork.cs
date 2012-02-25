/*
 * 
 * (c) Oleksandr Babii 2012
 * 
 */

using System;
using System.Collections.Generic;
using System.Linq;

namespace BrysonNet
{
    public class NeuralNetwork
    {
        #region Private fields

        private double _bias;
        private Random _random;

        #region Neuron and layer counts

        private readonly int _inputNeuronCount;
        private readonly int _outputNeuronCount;
        private readonly int[] _hiddenNeuronCount;
        private readonly int _hiddenLayerCount;

        #endregion

        private Dictionary<int, double[,]> _weight;
        private Dictionary<int, double[,]> _weightChange;
        private Dictionary<int, double[]> _signal;
         
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

        #endregion

        #region Constructors

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

            #region Signals initialization

            _signal = new Dictionary<int, double[]>(2 + _hiddenLayerCount); //input + output + all hidden
            _signal.Add(0, new double[_inputNeuronCount]);
            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                _signal.Add(i+1, new double[_hiddenNeuronCount[i]]);
            }
            _signal.Add(_signal.Count, new double[_outputNeuronCount]);

            #endregion
        }

        /// <summary>
        /// Pulsing the network.
        /// </summary>
        public void Pulse()
        {
            RandomizeWeights();
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
        }

        /// <summary>
        /// Training the network with specified inputs and desired outputs.
        /// </summary>
        /// <param name="input">Input values.</param>
        /// <param name="desiredOutput">Desired output values.</param>
        /// <param name="learningRate">Learning rate.</param>
        public void Train(double[][] input, double[][] desiredOutput, double learningRate)
        {
            if (input.GetLength(0) != _inputNeuronCount || desiredOutput.GetLength(0) != _outputNeuronCount)
                throw new ArgumentException();
        }

        private void RandomizeWeights()
        {
            _bias = _random.NextDouble();

            for (int layer = 0; layer < _hiddenLayerCount + 1; layer++)
            {
                for (int i = 0; i < _weight[layer].GetLength(0); i++)
                {
                    for (int j = 0; j < _weight[layer].GetLength(1); j++)
                    {
                        _weight[layer][i, j] = _random.NextDouble();
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

    }
}
