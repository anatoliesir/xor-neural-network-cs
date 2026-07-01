using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_C_
{
    public class MathFunctions
    {
        public static double SigmaForward(double[] prevLayer, double[][] weights, int to)
        {
            double sum = 0;
            for (int from = 0; from < prevLayer.Length; from++)
                sum += prevLayer[from] * weights[from][to];
            return sum;
        }
        public static double SigmaBackward(double[] nextLayer, double[][] weights, int to)
        {
            double sum = 0;
            for (int from = 0; from < nextLayer.Length; from++)
                sum += nextLayer[from] * weights[to][from];
            return sum;
        }
        public static double PReLU(double x, double alpha) // Hidden layer activation function(alpha is learnable)
        {
            return x >= 0 ? x : alpha * x;
        }
        public static double Sigmoid(double x)// Output layer activation function(the result is 0 to 1)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }
        public static double MSE(double output, double target) // I will use that to see if my neural network is training correctly
        {
            return Math.Pow(target - output, 2);
        }
    }
}
