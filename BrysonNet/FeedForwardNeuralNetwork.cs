/*
 * (c) Oleksandr Babii, 2012
 */

using System;
using BrysonNet.ActivationFunctions;
using BrysonNet.Training;

namespace BrysonNet
{
    public class FeedForwardNeuralNetwork : NeuralNetwork
    {
        private IFeedForwardTraining _training;

        /// <summary>
        /// Training type.
        /// </summary>
        public IFeedForwardTraining TrainingType
        {
            get { return (IFeedForwardTraining) _training; }
            set { _training = value; }
        }

        public new IFeedForwardActivation ActivationFunction
        {
            get { return (IFeedForwardActivation) _activation; }
            set { _activation = value; }
        }

        #region Constructors

        public FeedForwardNeuralNetwork()
        {
        }

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
        public FeedForwardNeuralNetwork(int inputNeuronCount, int hiddenNeuronCount1, int hiddenNeuronCount2,
                                        int outputNeuronCount)
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
        public FeedForwardNeuralNetwork(int inputNeuronCount, int hiddenNeuronCount1, int hiddenNeuronCount2,
                                        int hiddenNeuronCount3, int outputNeuronCount)
            : this(inputNeuronCount, hiddenNeuronCount1, hiddenNeuronCount2, outputNeuronCount)
        {
            _hiddenLayerCount = 3;
            _hiddenNeuronCount = new[] {hiddenNeuronCount1, hiddenNeuronCount2, hiddenNeuronCount3};
            _neurons = new[]
                {_inputNeuronCount, hiddenNeuronCount1, hiddenNeuronCount2, hiddenNeuronCount3, outputNeuronCount};
        }

        #endregion

        /// <summary>
        /// Training the network with specified inputs and desired outputs.
        /// </summary>
        /// <param name="input">Input values.</param>
        /// <param name="desiredOutput">Desired output values.</param>
        public override void Train(double[][] input, double[][] desiredOutput)
        {
            if (_training == null)
                _training = new BackPropagation();

            _training.Train(this, input, desiredOutput);
        }
    }
}
