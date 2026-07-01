using System.Collections.Specialized;
using System.Net;
using System.Net.WebSockets;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;


namespace Neural_Network_C_
{
    internal class Program
    {
        static public void Test(NeuralNetwork network)
        {
            // Nodes:
            Console.WriteLine("Node test:");
            for (int L = 0; L < network.layer.Count; L++)
            {
                for (int j = 0; j < network.layer[L].Length; j++)
                {
                    Console.Write(network.layer[L][j].ToString("F4") + " ");
                }
                Console.WriteLine();
            }

            // Biases:
            Console.WriteLine("\nBias test:");
            for (int L = 0; L < network.bias.Count; L++)
            {
                if (network.bias[L] != null)
                {
                    for (int j = 0; j < network.bias[L].Length; j++)
                    {
                        Console.Write(network.bias[L][j].ToString("F4") + " ");
                    }
                }
                Console.WriteLine();
            }

            // Alpha:
            Console.WriteLine("\nAlpha test:");
            for (int L = 0; L < network.alpha.Count; L++)
            {
                if (network.alpha[L] != null)
                    for (int j = 0; j < network.alpha[L].Length; j++)
                    {
                        Console.Write(network.alpha[L][j].ToString("F4") + " ");
                    }
                Console.WriteLine();
            }

            // Weights:
            Console.WriteLine("\nWeight test:");
            for (int L = 0; L < network.weight.Count; L++)
            {
                for (int i = 0; i < network.weight[L].Length; i++)
                {
                    for (int j = 0; j < network.weight[L][i].Length; j++)
                    {
                        Console.Write(network.weight[L][i][j].ToString("F4") + " ");
                    }
                }
                Console.WriteLine();
            }
        }
        static Random random = new Random();
        public static void InputRandomizer(ref double[] input, int min, int max)
        {
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = random.Next(min, max + 1);
            }
        }
        public static int FindInputTargetIndex(double[] userInput, int[,] inputTarget)
        {
            for (int i = 0; i < inputTarget.GetLength(0); i++)
            {
                bool ver = true;
                for (int j = 0; j < userInput.Length; j++)
                {
                    if (userInput[j] != inputTarget[i, j])
                    {
                        ver = false;
                        break;
                    }
                }
                if (ver) return i;
            }
            return -1;
        }

        // Functii de verificare:)
        public static int GetValidInt()
        {
            int result;
            while (!int.TryParse(Console.ReadLine(), out result))
            {
                Console.WriteLine("Error: Please enter a valid integer!!");
            }
            return result;
        }
        public static double GetValidDouble()
        {
            double result;
            while (!double.TryParse(Console.ReadLine(), out result))
            {
                Console.WriteLine("Error: Please enter a valid double/integer!!");
            }
            return result;
        }




        // VARIABILE GLOBALE OBLIGATORII!!
        // Initializez toate variebilele necesare:
        static NeuralNetwork network = new NeuralNetwork();
        static double learningRate = 0.05;
        static int nrEpoch = 0;

        // Poti modifica sistema pe care se antreneaza robotul
        static int[,] inputTarget = new int[4, 3] { { 0, 0, 0 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 0 } };// XOR

        // temporar lists:
        static List<double[]> z = new List<double[]>(); // Pre-Activation function value
        static List<double[]> unitError = new List<double[]>(); // Unit Error of each node during backpropagation
        static double[] userInput;

        static bool verLoop = true;
        static void Main(string[] args)
        {
            NeuralNetwork.Initialize(network, out userInput, z, unitError);

            network = JsonSave.Load();
            while (verLoop)
            {
                MainLogic();
                Console.WriteLine("---------------------------------------------------------------------------\n");

            }
            Console.WriteLine("Thank you for using our neural network(XOR)!!:)");

        }



        private static void MainLogic()
        {
            Console.WriteLine("Welcome to a XOR logic gate neural network!\nSelect option:\n1. Use neural network\n2. Train neural network\n3. Reset neural network\n4. Exit\n" +
                "5. Show/Test neural network memory");
            int useTrainOption = GetValidInt();

            if (useTrainOption == 1)// Use option
            {
                Console.WriteLine("Insert the values of the 2 input nodes(from 0 to 1):");
                for (int i = 0; i < network.layer[0].Length; i++)
                {
                    network.layer[0][i] = GetValidInt();
                    userInput[i] = network.layer[0][i];
                }
                int target = -1;
                try
                {
                    target = inputTarget[FindInputTargetIndex(userInput, inputTarget), 2];
                }
                catch (Exception e)
                {
                    Console.WriteLine("!!!!!!!!!!!!Enter only 0 or 1!!!!!!!!!!!");
                    return;
                }

                // Forwardpropagation:
                // PReLU activation function:
                for (int L = 0; L < network.layer.Count - 2; L++)
                {
                    for (int to = 0; to < network.layer[L + 1].Length; to++)
                    {
                        z[L + 1][to] = MathFunctions.SigmaForward(network.layer[L], network.weight[L], to) + network.bias[L + 1][to];
                        network.layer[L + 1][to] = MathFunctions.PReLU(z[L + 1][to], network.alpha[L + 1][to]);

                    }
                }
                // Sigmoid activation function:
                for (int to = 0; to < network.layer[network.layer.Count - 1].Length; to++)
                {
                    z[network.layer.Count - 1][to] = MathFunctions.SigmaForward(network.layer[network.layer.Count - 2], network.weight[network.layer.Count - 2], to) + network.bias[network.layer.Count - 1][to];
                    network.layer[network.layer.Count - 1][to] = MathFunctions.Sigmoid(z[network.layer.Count - 1][to]);
                }
                Console.WriteLine($"Output = {network.layer[network.layer.Count - 1][0]:F4}\nTarget = {target}");
                
            }

            else if (useTrainOption == 2)// Train option
            {
                Console.WriteLine("Insert how many epochs do you want the neural network to train:");
                nrEpoch = GetValidInt();
                int contorTemp = 0;

                // For loop for training
                for (int epoch = 0; epoch < nrEpoch; epoch++)
                {
                    // XOR input randomizer:
                    double[] randomizedInputs = new double[network.layer[0].Length];
                    InputRandomizer(ref randomizedInputs, 0, 1);
                    network.layer[0] = randomizedInputs;
                    int target = inputTarget[FindInputTargetIndex(randomizedInputs, inputTarget), 2];

                    // Forwardpropagation:
                    // PReLU activation function:
                    for (int L = 0; L < network.layer.Count - 2; L++)
                    {
                        for (int to = 0; to < network.layer[L + 1].Length; to++)
                        {
                            z[L + 1][to] = MathFunctions.SigmaForward(network.layer[L], network.weight[L], to) + network.bias[L + 1][to];
                            network.layer[L + 1][to] = MathFunctions.PReLU(z[L + 1][to], network.alpha[L + 1][to]);

                        }
                    }
                    // Sigmoid activation function:
                    for (int to = 0; to < network.layer[network.layer.Count - 1].Length; to++)
                    {
                        z[network.layer.Count - 1][to] = MathFunctions.SigmaForward(network.layer[network.layer.Count - 2], network.weight[network.layer.Count - 2], to) + network.bias[network.layer.Count - 1][to];
                        network.layer[network.layer.Count - 1][to] = MathFunctions.Sigmoid(z[network.layer.Count - 1][to]);
                    }

                    // Backpropagation:
                    // Output layer unitError = activationFunction*(1 - activationFunction)*(target - activationFunction)
                    for (int to = 0; to < network.layer[network.layer.Count - 1].Length; to++)
                    {

                        // Formula originala pentru calcularea unitatii de eroare folosind derivata lui sigmoid + MSE
                        unitError[unitError.Count - 1][to] = network.layer[network.layer.Count - 1][to] *
                            (1 - network.layer[network.layer.Count - 1][to]) *
                            (target - network.layer[network.layer.Count - 1][to]);
                    }
                    // Hidden layer unitError = activationFunction*(1 - activationFunction)*Sigma(unitError_next * weight_j_next)
                    for (int L = network.layer.Count - 2; L > 0; L--)
                    {
                        for (int to = 0; to < network.layer[L].Length; to++)
                        {
                            double dAct = (z[L][to] >= 0) ? 1.0 : network.alpha[L][to]; // Derivative of the PReLU activation function
                            unitError[L][to] = dAct * MathFunctions.SigmaBackward(unitError[L + 1], network.weight[L], to);
                        }
                    }

                    // Weight update_i_j += learningRate * unitError_j * activationFunction
                    // Bias update_j += learningRate * unitError_j
                    // Alpha update_j += learningRate * unitError_j * preActivationFunction  IF 
                    for (int L = 1; L < network.layer.Count; L++)
                    {
                        for (int to = 0; to < network.layer[L].Length; to++)
                        {
                            for (int from = 0; from < network.layer[L - 1].Length; from++)
                            {
                                network.weight[L - 1][from][to] += learningRate * unitError[L][to] * network.layer[L - 1][from];
                            }
                            network.bias[L][to] += learningRate * unitError[L][to];
                            if (L < network.layer.Count - 1 && z[L][to] < 0)
                                network.alpha[L][to] += learningRate * unitError[L][to] * z[L][to];
                        }
                    }

                    Console.WriteLine($"Target = {target}" +
                        $"\nOutput = {network.layer[network.layer.Count - 1][0].ToString("F4")}" +
                        $"\nMSE = {MathFunctions.MSE(network.layer[network.layer.Count - 1][0], target).ToString("F4")}\n");

                    if (MathFunctions.MSE(network.layer[network.layer.Count - 1][0], target) <= 0.0001)
                        contorTemp++;
                    if (MathFunctions.MSE(network.layer[network.layer.Count - 1][0], target) >= 0.05) 
                        contorTemp = 0;
                    if (contorTemp == 100)
                    {
                        Console.WriteLine("The neural network is trained successfully because it has a very low Mean Squared Error(MSE)!!!");
                        break;
                    }
                        
                }

                Console.WriteLine("\nSave?\n1. yes\n2. no");
                int saveOption = GetValidInt();
                if (saveOption == 1) JsonSave.Save(network);
                //Test(network);
                
            }
            else if (useTrainOption == 3) // Reset neural network
            {
                network = null;
                network = new NeuralNetwork();

                network.SetInput(2);
                network.SetHidden(4, 3);
                network.SetOutput(1);
                network.SetWeights();

                z.Clear();
                unitError.Clear();
                userInput = new double[network.layer[0].Length];
                for (int L = 0; L < network.layer.Count; L++)
                {
                    z.Add(new double[network.layer[L].Length]);
                    unitError.Add(new double[network.layer[L].Length]);
                }
                JsonSave.Save(network);
                Console.WriteLine("Neural network resetted successfully!");
                
            }
            else if (useTrainOption == 4) // Exit the application
            {
                verLoop = false;                
                return;
            }
            else if(useTrainOption == 5)
            {
                Test(network);                
            }
            else
            {
                Console.Clear();
                Console.WriteLine("Please enter a valid option!");
            }
        }
    }
}
