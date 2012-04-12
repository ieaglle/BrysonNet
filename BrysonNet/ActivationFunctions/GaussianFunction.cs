using System;

namespace BrysonNet.ActivationFunctions
{
    public class GaussianFunction : IRadialBasisActivation
    {
        public double Calc(double net)
        {
            return Math.Exp(-(net*net));
        }

        public double Derivative(double signal)
        {
            return -2 * signal*Math.Exp(-(signal * signal));
        }
    }
}
