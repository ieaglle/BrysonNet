using System;
using System.Diagnostics;

namespace BrysonNet.Training
{
    public class AdaptiveBackPropagation : IFeedForwardTraining
    {
        private double _learningRate;
        private readonly double _precision;
        private double[] _lr;
        private readonly double _adaptationRate;
        private readonly double _momentumRate ;

        public AdaptiveBackPropagation()
        {
            _learningRate = 1.0;
            _precision = 0.01;
            _adaptationRate = 0.5;
            _momentumRate = 0.5;
        }

        /// <summary>
        /// This type of training changes learning rate to achieve faster network convergence. 
        /// </summary>
        /// <param name="learningRate">Learning rate of training (it is only starting point, it will be changed automatically).</param>
        /// <param name="precision">Precision</param>
        /// <param name="adaptationRate">How fast learning rate is adapting.</param>
        /// <param name="momentumRate"><para>Momentum is used to prevent from finding local minima.</para>Range: [0, 1].</param>
        public AdaptiveBackPropagation(double learningRate, double precision, double momentumRate, double adaptationRate)
        {
            _learningRate = learningRate;
            _precision = precision;
            _adaptationRate = adaptationRate;
            _momentumRate = momentumRate;
        }

        public void Train(FeedForwardNeuralNetwork net, double[][] input, double[][] desiredOutput)
        {
            int inputNeuronCount = net.InputSignal.Length;
            int outputNeuronCount = net.OutputSignal.Length;

            if (input[0].Length != inputNeuronCount || desiredOutput[0].Length != net.OutputSignal.Length)
                throw new Exception("Input or output values number are invalid.");

            // maximum allowed epoches
            const int maxEpoches = 500000000;
            bool go = true;
            //number of learning sequences
            int sequences = input.Length;
            double[][] signal = net.Signal;
            double[][] error = net.Error;
            double[][][] weight = net.Weight;
            double[][][] weightChange = net.WeightChange;
            int[] neurons = net.Neurons;
            int layerCount = net.Layers;

            double previousAverageError = 0;
            while (go)
            {
                // check for infinite loop
                if (net.Epoch >= maxEpoches)
                    throw new Exception("Training takes too long. Try to change network architecture.");

                double averageError=0;
                for (int curr = 0; curr < sequences; curr++)
                {
                    double[] currInput = input[curr];
                    double[] currDesiredOutput = desiredOutput[curr];
                    //setting input signals
                    net.InputSignal = currInput;
                    //executing network
                    net.Pulse();
                    //calculating errors of output neurons 
                    for (int i = 0; i < outputNeuronCount; i++)
                    {
                        error[layerCount - 1][i] =
                            net.ActivationFunction.Derivative(signal[layerCount - 1][i]) *
                            (currDesiredOutput[i] - signal[layerCount - 1][i]);
                        averageError += 0.5f * (currDesiredOutput[i] - signal[layerCount - 1][i]) *
                                        (currDesiredOutput[i] - signal[layerCount - 1][i]);
                    }
                    _learningRate = _learningRate * (averageError >= previousAverageError ? 0.2 : 5); //_learningRate * (_adaptationRate * averageError * previousAverageError + 1);
                    previousAverageError = averageError;

                    //calculating all errors and wight changes
                    for (int layer = layerCount - 1; layer > 0; layer--)
                    {
                        for (int i = 0; i < neurons[layer - 1]; i++)
                        {
                            double temp = 0;
                            for (int j = 0; j < neurons[layer]; j++)
                            {
                                double delta = _learningRate * error[layer][j] * signal[layer - 1][i];
                                //adding momentum to prevent from detecting a local minima
                                weight[layer - 1][i][j] += delta + _momentumRate * weightChange[layer - 1][i][j];
                                weightChange[layer - 1][i][j] = delta;
                                temp += error[layer][j] * weight[layer - 1][i][j];
                            }
                            error[layer - 1][i] = net.ActivationFunction.Derivative(signal[layer - 1][i]) * temp;
                        }
                    }
                }
                
                if (averageError <= _precision)
                    go = false;
            }
        }
    }
}


//_learningRate = _learningRate*(_adaptationRate*averageError*previousAverageError + 1);
                    //_lr[curr] = _lr[curr] * (_adaptationRate * averageError * previousAverageError + 1);
//previousAverageError = averageError;