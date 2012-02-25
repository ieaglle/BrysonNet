using System;

namespace BrysonNet
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork net = new NeuralNetwork(2, 2000, 280, 25, 2);
            net.Initialize();

            net.InputSignal[0] = 2.0;
            net.InputSignal[1] = 5.0;

            net.Pulse();
            net.Show();

            Console.Read();
        }
    }
}
