using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
namespace Neural_Network_C_
{    
    public class NeuralNetwork
    {
        static Random random = new Random();
        static double GetRandomNumber(double minimum, double maximum) // Creez randomizari de tip double(min, max)
        {
            return random.NextDouble() * (maximum - minimum) + minimum; // Prima data aud de formula data
        }
        public List<double[]> layer = new List<double[]>();
        public List<double[]> bias = new List<double[]>();
        public List<double[][]> weight = new List<double[][]>();
        public List<double[]> alpha = new List<double[]>();

        // Etapele crearii unui sistem de noduri si bias:
        public void SetInput(int n) // Prima etapa
        {
            layer.Add(new double[n]);
            bias.Add(null);
            alpha.Add(null);
        }
        public void SetHidden(int n, int m) // A doua etapa
        {
            for (int i = 0; i < m; i++)
            {
                layer.Add(new double[n]);
                bias.Add(new double[n]);
                alpha.Add(Enumerable.Repeat(0.01, n).ToArray());
            }
        }
        public void SetOutput(int n) // A treia etapa
        {
            layer.Add(new double[n]);
            bias.Add(new double[n]);
            alpha.Add(null);
        }        
        public void SetWeights() // A patra etapa
        {
            for (int i = 0; i < layer.Count - 1; i++)
            {
                double[][] w = new double[layer[i].Length][];
                for (int r = 0; r < layer[i].Length; r++)
                {
                    w[r] = new double[layer[i + 1].Length];
                }
                weight.Add(w);
            }

            // Randomizez greutatile
            for (int L = 0; L < weight.Count; L++)
            {
                for (int i = 0; i < weight[L].Length; i++)
                {
                    for (int j = 0; j < weight[L][i].Length; j++)
                    {
                        weight[L][i][j] = GetRandomNumber(-1, 1);
                    }
                }
            }
        }


        public static void Initialize(NeuralNetwork network, out double[] userInput, List<double[]> z, List<double[]> unitError)
        {
            network.SetInput(2);
            network.SetHidden(4, 3);
            network.SetOutput(1);
            network.SetWeights();

            userInput = new double[network.layer[0].Length];

            z.Clear();
            unitError.Clear();

            for (int L = 0; L < network.layer.Count; L++)
            {
                z.Add(new double[network.layer[L].Length]);
                unitError.Add(new double[network.layer[L].Length]);
            }
        }

    }
}
