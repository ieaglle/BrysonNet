/*
 * Oleksandr Babii, 2012
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrysonNet
{
    public class KohonenSelfOrganisingFeatureMap
    {
        private int _inputNeuronCount;
        private int _mapSizeX;
        private int _mapSizeY;
        private double[][][] _weight;
        private Random _random;
        private double[][][] _distance;
        private int _radius;
        private int _radiusCurr;
        private const int _radiusChange = -1;
        private double _learningRate = 0.9;
        private double _learningRateCurr;
        private const double _learningRateChange = 0.8;
        private BestMatchingUnit _bmu;
        private int _epoch;

        #region Public Properties

        /// <summary>
        /// Current epoch of the network.
        /// </summary>
        public int Epoch
        {
            get { return _epoch; }
        }

        /// <summary>
        /// Current best matching unit.
        /// </summary>
        public BestMatchingUnit BestMatchingUnit
        {
            get { return _bmu; }
        }

        #endregion

        //public KohonenSelfOrganisingFeatureMap() {}

        public KohonenSelfOrganisingFeatureMap(int inputNeuronCount, int mapSizeX, int mapSizeY)
        {
            _inputNeuronCount = inputNeuronCount;
            _mapSizeX = mapSizeX;
            _mapSizeY = mapSizeY;
        }

        public void Initialize()
        {
            _radius = Math.Max(_mapSizeX, _mapSizeY)/2;
            _random = new Random();
            _weight = new double[_mapSizeX][][];
            _distance = new double[_mapSizeX][][];

            for (int i = 0; i < _mapSizeX; i++)
            {
                _weight[i] = new double[_mapSizeY][];
                _distance[i] = new double[_mapSizeY][];
                for (int j = 0; j < _mapSizeY; j++)
                {
                    _weight[i][j] = new double[_inputNeuronCount];
                    _distance[i][j] = new double[1];
                }
            }
        }

        public void RandomizeWeights(double minValue = 0, double maxValue = 1)
        {
            if (maxValue <= minValue)
                throw new Exception("Maximum has to be greater then minimum, didn't you know?");

            for (int i = 0; i < _mapSizeX; i++)
            {
                for (int j = 0; j < _mapSizeY; j++)
                {
                    for (int k = 0; k < _inputNeuronCount; k++)
                    {
                        _weight[i][j][k] = _random.NextDouble()*(maxValue - minValue) + minValue;
                    }
                }
            }
        }

        public void Train(double[] input)
        {
            bool go = true;

            while (go)
            {
                CalculateDistances(input);

                GetBestMatchingUnit();

                int currRadius = CalculateRadius();

                for (int i = -currRadius; i < currRadius; i++)
                {
                    for (int j = -currRadius; j < currRadius; j++)
                    {
                        //Math.Min(_bmu.X + i, )
                    }
                }

                _epoch++;
                go = false;
            }
        }

        private void CalculateDistances(double[] input)
        {
            if (input.Length != _inputNeuronCount)
                throw new ArgumentOutOfRangeException();

            for (int i = 0; i < _mapSizeX; i++)
            {
                for (int j = 0; j < _mapSizeY; j++)
                {
                    double distance = 0;
                    for (int k = 0; k < _inputNeuronCount; k++)
                    {
                        distance += (input[k] - _weight[i][j][k])*(input[k] - _weight[i][j][k]);
                    }
                    _distance[i][j][0] = Math.Sqrt(distance);
                }
            }
        }

        private void GetBestMatchingUnit()
        {
            double minimumValue = Double.MaxValue;
            for (var i = 0; i < _mapSizeX; i++)
            {
                for (var j = 0; j < _mapSizeY; j++)
                {
                    var current = _distance[i][j][0];
                    if (current <= minimumValue)
                    {
                        minimumValue = current;
                        _bmu.X = i;
                        _bmu.Y = j;
                    }
                }
            }
        }

        private void CalculateLearningRate()
        {
            if (_learningRateCurr > 0.1)
            {
                _learningRateCurr = _learningRate - 0.01*_epoch;
            }  
        }

        private int CalculateRadius()
        {
            if (_radiusCurr > 1)
            {
                _radiusCurr = _radius - _epoch;
                return _radiusCurr;
            }
            return 1;
        }


    }
}
