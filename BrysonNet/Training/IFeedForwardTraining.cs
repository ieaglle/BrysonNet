namespace BrysonNet.Training
{
    public interface IFeedForwardTraining : ITraining
    {
        void Train(FeedForwardNeuralNetwork net, double[][] input, double[][] desiredOutput);
    }
}
