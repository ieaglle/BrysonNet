namespace BrysonNet.ActivationFunctions
{
    public interface IFeedForwardActivation : IActivationFunction
    {
        double Derivative(double signal);
    }
}
