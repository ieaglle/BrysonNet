using System;

namespace BrysonNet.ActivationFunctions
{
    public class BipolarSigmoid : IFeedForwardActivation
    {
        public double Calc(double net)
        {
            var k = Math.Exp(-net);
            return 2 / (1.0f + k) - 1;
        }

        public double Derivative(double signal)
        {
            return 0.5f * (1.0f + signal)*(1.0f - signal);
        }
    }
}
