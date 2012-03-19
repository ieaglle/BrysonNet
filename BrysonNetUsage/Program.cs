using System;
using BrysonNet;
using BrysonNet.ActivationFunctions;

namespace BrysonNetUsage
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork(2, 20, 1);
            net.Initialize();
            net.RandomizeWeights(-.5, .5);
            net.ActivationFunction = new BipolarSigmoid();

            const double high = .9;
            const double low = -.9;

            double[][] input = new[]
                                   {
                                       new[] {low, low},
                                       new[] {low, high},
                                       new[] {high, low},
                                       new[] {high, high}
                                   };
            double[][] output = new[]
                                   {
                                       new[] {low},
                                       new[] {high},
                                       new[] {high},
                                       new[] {low}
                                   };

            DateTime start = DateTime.Now;
            net.Train(input, output, 0.00000001);

            TimeSpan dur = DateTime.Now - start;
            Console.WriteLine(dur);
            //net.Save("XOR.xml");
            
            Console.Out.WriteLine("\nPassed epoches: " + net.Epoch);
            
            net.InputSignal = new[] { low, low };
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] { low, high };
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] { high, low };
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] { high, high };
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);
            
            /*
            FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork();
            net.Load("AND.xml");

            net.InputSignal = new[] {.1, .1};
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] {.1, .9};
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] {.9, .1};
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] {.9, .9};
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);
            */

            Console.Read();
        }
    }
}
