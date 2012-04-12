namespace BrysonNet.ActivationFunctions
{
    public class StepFunction : IActivationFunction
    {
        private readonly double _threshold;

        public StepFunction(double threshold)
        {
            _threshold = threshold;
        }

        public double Calc(double net)
        {
            return net > _threshold ? 0.9 : -0.9;
        }
    }
}
