using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Neural_Network_C_
{
    public class JsonSave // Trebuie sa invat o zi cum corect sa salvez in fisier Json :(
    {
        private static readonly JsonSerializerOptions Options = new JsonSerializerOptions
        {
            WriteIndented = true,
            IncludeFields = true
        };

        private static readonly string FolderPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "NeuralNetworkRazvan"
        );

        private static readonly string FilePath = Path.Combine(FolderPath, "NeuralNetworkJson.json");

        public static void Save(NeuralNetwork network)
        {
            if (!Directory.Exists(FolderPath))
            {
                Directory.CreateDirectory(FolderPath);
            }

            string json = JsonSerializer.Serialize(network, Options);
            File.WriteAllText(FilePath, json);
        }

        public static NeuralNetwork Load()
        {
            if (!File.Exists(FilePath))
            {
                Console.WriteLine("Fisierul JSON nu a fost gasit. Se creeaza unul nou:)\n");
                NeuralNetwork newNetwork = new NeuralNetwork();

                double[] userInput;
                List<double[]> z = new List<double[]>();
                List<double[]> unitError = new List<double[]>();

                NeuralNetwork.Initialize(newNetwork, out userInput, z, unitError);

                Save(newNetwork);

                return newNetwork;
            }

            string json = File.ReadAllText(FilePath);
            return JsonSerializer.Deserialize<NeuralNetwork>(json, Options);
        }
    }        
}

