using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BrysonNet
{
    public class FeedForwardRadialBasisNeuralNetwork : INeuralNetwork
    {
        public void Initialize()
        {
            throw new NotImplementedException();
        }

        public void Pulse()
        {
            throw new NotImplementedException();
        }

        public void Save(string filename)
        {
            throw new NotImplementedException();
        }

        public void Load(string filename)
        {
            throw new NotImplementedException();
        }


        private double GaussianFunction(double net)
        {
            return Math.Exp(-Math.Pow(net, 2));
        }
    }
}
