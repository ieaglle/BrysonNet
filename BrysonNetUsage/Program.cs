using System;
using BrysonNet;
using BrysonNet.ActivationFunctions;
using BrysonNet.Training;

namespace BrysonNetUsage
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            
            FeedForwardNeuralNetwork net = new FeedForwardNeuralNetwork(2, 3, 1);
            net.Initialize();
            net.RandomizeWeights(-1, 1);
            net.ActivationFunction = new BipolarSigmoid();
            net.TrainingType = new BackPropagation(0.1, 0.0001, 0.9);
            
            const double high = 0.9;
            const double low = -0.9;

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
            net.Train(input, output);
            
            TimeSpan dur = DateTime.Now - start;
            Console.WriteLine(dur.ToString());
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
            

            
            /*
            int[][] input2 = new[]
                {
                    new[] {1, -1, 1, 1}
                    //new[] {1, -1, 1, -1}
                };
            HopfieldNeuralNetwork hnn = new HopfieldNeuralNetwork(input2);
            hnn.Initialize();
            hnn.Check(new[] { -1, -1, 1, 1 });
            hnn.Show();*/
            
            //Console.Read();
            /*

            KohonenSelfOrganisingFeatureMap som = new KohonenSelfOrganisingFeatureMap(3, 1000, 1000);
            som.Initialize();
            som.RandomizeWeights();
            som.Train(new[] {0.4, 5.0, 7.02});
            //som.CalculateDistances(new[] {0.1, 0.4, 0.9});
            //DateTime start = DateTime.Now;
            //som.GetBestMatchingUnit();
            //DateTime end = DateTime.Now;
            //var dur = end - start;
            //Console.WriteLine("Time is: {0}",dur.TotalMilliseconds);
            */
            Console.Read();

        }
    }
}
