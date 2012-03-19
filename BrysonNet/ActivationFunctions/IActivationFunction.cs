namespace BrysonNet.ActivationFunctions
{
    public interface IActivationFunction
    {
        double Calc(double net);
        double Derivative(double signal);
    }
}
