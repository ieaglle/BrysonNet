using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using BrysonNet.ActivationFunctions;
using System.Diagnostics;

namespace BrysonNet
{
    public class HopfieldNeuralNetwork
    {
        private int[] _input;
        private int[][] _pattern;
        private int[][] _weight;
        private Random _random;
        private readonly int _neuronCount;
        private Signum _signum;

        public HopfieldNeuralNetwork(int[][] pattern)
        {
            _random = new Random(DateTime.Now.Millisecond);
            _signum = new Signum();
            _neuronCount = pattern[0].Length;
            _pattern = pattern;
            _input = new int[_neuronCount];
        }

        public void Initialize()
        {
            _weight = new int[_neuronCount][];
            for (int i = 0; i < _neuronCount; i++)
            {
                _weight[i] = new int[_neuronCount];
            }
            SetWeights();
        }

        private void SetWeights()
        {
            for (int i = 0; i < _neuronCount; i++)
            {
                
                for (int j = 0; j < _neuronCount; j++)
                {
                    int v = 0;
                    for (int index = 0; index < _pattern.Length; index++)
                    {
                        v += _pattern[index][i]*_pattern[index][j];
                        _weight[i][j] = i != j ? v : 0;
                    }
                }
            }
        }

        private bool Pulse()
        {
            int neuron = (int) (_random.NextDouble()*_neuronCount);
            Debug.WriteLine(neuron);
            int net = 0;

            for (int i = 0; i < _neuronCount; i++)
            {
                net += _input[i] * _weight[i][neuron];
            }
            if (net <= 0)
            {
                _input[neuron] = _input[neuron] * _signum.Calc(net);
                return false;
            }
            return true;
        }

        public void Check(int[] input)
        {
            _input = input;
            bool go = true;
            while (go)
            {
                bool curr = Pulse();
                //Debug.WriteLine(curr);
                go = go && !curr;

            }
        }

        public void Show()
        {
            for (int i = 0; i < _neuronCount; i++)
            {
                Console.Write("{0}  ", _input[i]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < _neuronCount; i++)
            {
                for (int j = 0; j < _neuronCount; j++)
                {
                    Console.Write("{0}\t", _weight[i][j]);
                }
                Console.WriteLine();
            }
        }


    }
}
