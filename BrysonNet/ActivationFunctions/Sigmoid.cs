using System;

namespace BrysonNet.ActivationFunctions
{
    public class Sigmoid : IFeedForwardActivation
    {
        public double Calc(double net)
        {
            var k = Math.Exp(-net);
            return 1 / (1.0f + k);
        }

        public double Derivative(double signal)
        {
            return signal*(1.0f - signal);
        }
    }
}
