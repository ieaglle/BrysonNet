using System;
using System.Globalization;
using BrysonNet;

namespace BrysonNetUsage
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            NeuralNetwork net = new NeuralNetwork(2, 2, 1);
            net.Initialize();
            net.RandomizeWeights();

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
            net.Train(input, output, 0.0001);
            net.Save("XOR.xml");
            
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
            */
            
            NeuralNetwork net = new NeuralNetwork();
            net.Load("XOR.xml");

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
            
            Console.Read();
        }
    }
}
