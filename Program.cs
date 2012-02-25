using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BrysonNet
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork net = new NeuralNetwork(2, 2000, 280, 25, 2);
            net.Initialize();

            net.InputSignal[0] = 2.0;
            //net.InputSignal[1] = 5.0;

            net.Pulse();
            //net.Show();

            double[][] input = new[]
                                   {
                                       new double[3],
                                       new double[1], 
                                   };

            double[][] output = new[]
                                    {
                                        new double[5],
                                        new double[5], 
                                    };

            net.Train(input, output, 3.0);

            Console.Read();
        }
    }
}
