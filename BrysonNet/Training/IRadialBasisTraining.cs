namespace BrysonNet.Training
{
    public interface IRadialBasisTraining : ITraining
    {
        void Train(RadialBasisNeuralNetwork net, double[][] input, double[][] desiredOutput);
    }
}
