using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using BrysonNet.ActivationFunctions;
using BrysonNet.Training;

namespace BrysonNet
{
    public class RadialBasisNeuralNetwork : NeuralNetwork
    {
        public new IRadialBasisActivation ActivationFunction
        {
            get { return (IRadialBasisActivation) _activation; }
            set { _activation = value; }
        }

        private IRadialBasisTraining _training;

        /// <summary>
        /// Training type.
        /// </summary>
        public IRadialBasisTraining TrainingType
        {
            get { return _training; }
            set { _training = value; }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputNeuronCount">Number of input neurons.</param>
        /// <param name="hiddenNeuronCount">Number of hidden neurons.</param>
        /// <param name="outputNeuronCount">Number of output neurons.</param>
        public RadialBasisNeuralNetwork(int inputNeuronCount, int hiddenNeuronCount, int outputNeuronCount)
        {
            _inputNeuronCount = inputNeuronCount;
            _hiddenLayerCount = 1;
            _hiddenNeuronCount = new[] {hiddenNeuronCount};
            _neurons = new[] {inputNeuronCount, hiddenNeuronCount, outputNeuronCount};
        }

        public override void Initialize()
        {
            //base.Initialize();
            _random = new Random();
            _epoch = 0;
        }

        public override void Train(double[][] input, double[][] desiredOutput)
        {
            _training.Train(this, input, desiredOutput);
        }
    }
}
