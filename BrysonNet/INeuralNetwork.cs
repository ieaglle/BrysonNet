using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BrysonNet
{
    interface INeuralNetwork
    {
        void Initialize();
        void Pulse();
        void Save(string filename);
        void Load(string filename);
    }
}
