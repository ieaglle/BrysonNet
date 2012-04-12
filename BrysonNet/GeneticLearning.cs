using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BrysonNet
{
    public class GeneticLearning
    {
        private int _chromosomesCount;
        private double _crossoverRate = 0.7;
        private double _mutationRate = 0.5;
        private Random _rand;

        public GeneticLearning()
        {
            _rand = new Random();
        }

        private double[][] _chromosomes;

        public void Crossover(double[] chromosome1, double[] chromosome2)
        {
            if (chromosome1.Length != chromosome2.Length)
                throw new ArgumentException("Chromosome lenghts are not equal.");

            if (_rand.NextDouble() < 0.7)
            {
                int length = chromosome1.Length;
                int position = (int)(_rand.NextDouble() * length);
                for (int curr = position; curr < length; curr++)
                {
                    double temp = chromosome1[curr];
                    chromosome1[curr] = chromosome2[curr];
                    chromosome2[curr] = temp;
                }
            }
        }

    }
}
