using System;
using System.Globalization;
using BrysonNet;

namespace BrysonNetUsage
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork(2, 40, 40, 1);
            net.Initialize();
            net.RandomizeWeights(-1);

            double[][] input = new[]
                                   {
                                       new[] {.1, .1},
                                       new[] {.1, .9},
                                       new[] {.9, .1},
                                       new[] {.9, .9}
                                   };
            double[][] output = new[]
                                   {
                                       new[] {.1},
                                       new[] {.9},
                                       new[] {.9},
                                       new[] {.1}
                                   };

            DateTime start = DateTime.Now;
            net.Train(input, output, 0.0001, .5);

            TimeSpan dur = DateTime.Now - start;
            Console.WriteLine(dur);
            //net.Save("XOR.xml");
            
            Console.Out.WriteLine("\nPassed epoches: " + net.Epoch);

            net.InputSignal = new[] { .1, .1 };
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] { .1, .9 };
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] { .9, .1 };
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);

            net.InputSignal = new[] { .9, .9 };
            net.Pulse();
            Console.Out.WriteLine("Output: {0}", net.OutputSignal[0]);
            
            
            /*FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork();
            net.Load("XOR.xml");

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
