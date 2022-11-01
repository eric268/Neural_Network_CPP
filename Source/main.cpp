#include "../Include/pch.h"
#include "../Include/MathHelper.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"

void ReadMNIST(std::string, std::string, std::vector<std::vector<double>>&, std::vector<int>&);
void DrawNumber(const std::vector<std::vector<double>>&, const int, const int, const int);
void UpdateBias(NetworkLayer*, LayerResults*);
void UpdateResults(NeuralNetwork&, double);
void RunNetwork(NeuralNetwork&, const double, const std::vector<std::vector<double>>, const std::vector<int>, bool);
void UpdateWeights(NetworkLayer*, LayerResults*);
void BeginTest(NeuralNetwork&, bool);
void SaveWeightsAndBias();

#define NumTraining 60'000
#define NumTesting 10'000
#define NumOfPixels 784
//#define DrawNumbers

int main()
{
    NeuralNetwork neuralNetwork(std::vector<int>{ 784, 16, 16, 10 });

    std::vector<std::vector<double>> imageArray(NumTraining, std::vector<double>(NumOfPixels));
    std::vector<int> labelArray(NumTraining, -1);
    const std::string trainingImagesPath = "MNISTData/train-images.idx3-ubyte";
    const std::string trainingLabelsPath = "MNISTData/train-labels.idx1-ubyte";
    ReadMNIST(trainingImagesPath, trainingLabelsPath, imageArray, labelArray);
    double testSize = 100.0;
    int numTests = 1000;
    int testCounter = 0;
    char c = ' ';
    while (true)
    {
        RunNetwork(neuralNetwork, testSize, imageArray, labelArray, true);
        testCounter++;
        if (testCounter >= numTests)
        {
            testCounter = 0;
            std::cout << "Press q to quit:\n";
            std::cout << "Press t to test:\n";
            std::cout << "Press s to save:\n";
            std::cout << "Press anything else to retrain\n";
            std::cin >> c;
            if (c == 'q' || c == 't' || c == 's')
                break;
        }
    }
    if (c == 't')
    {
        BeginTest(neuralNetwork, false);
    }
    else if (c == 's')
    {
        SaveWeightsAndBias();
    }
    return 0;
}

void SaveWeightsAndBias()
{
    std::string saveWeightName, saveBiasName;
    std::cout << "Enter save weight name: ";
    std::cin >> saveWeightName;
    std::cout << "Enter save bias name: ";
    std::cin >> saveBiasName;
    //TODO create output file functions
}

void BeginTest(NeuralNetwork& network, bool isTraining)
{
    const std::string testImagesPath = "MNISTData/t10k-images.idx3-ubyte";
    const std::string testLabelsPath = "MNISTData/t10k-labels.idx1-ubyte";
    std::vector<std::vector<double>> testImageArray(NumTesting, std::vector<double>(NumOfPixels));
    std::vector<int> testLabelArray(NumTesting, -1);

    ReadMNIST(testImagesPath, testLabelsPath, testImageArray, testLabelArray);
    RunNetwork(network, NumTesting, testImageArray, testLabelArray, isTraining);
}


void RunNetwork(NeuralNetwork& network, const double testSize, const std::vector<std::vector<double>> imageArray, const std::vector<int> labelArray, bool isTraining)
{
    char n = ' ';
    int counter = 0;
    int rightAnswers = 0;
    network.mTotalError = 0.0;

    for (int i = 0; i < network.mLayerResults.size(); i++)
    {
        std::fill(network.mLayerResults[i]->mBiasResults.begin(), network.mLayerResults[i]->mBiasResults.end(), 0.0);

        for (auto& p : network.mLayerResults[i]->mWeightedResults)
        {
            std::fill(p.begin(), p.end(), 0.0);
        }
    }
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, NumTraining - 1);
    while (counter < testSize)
    {
        int val = (isTraining) ? dist(rd) : counter;
        int correctAns = labelArray[val];
        int networkGuess = network.RunNetwork(imageArray[val]);

#ifdef DrawNumbers
        DrawNumber(imageArray, correctAns, val, networkGuess);
        std::cin >> n;
        if (n == 'q')
            break;
#endif
        counter++;
        if (networkGuess == correctAns)
            rightAnswers++;
        network.CalculateLayerDeltaCost(correctAns);
    }
    network.UpdateResults(testSize);
    std::cout << "Correct %" << std::to_string(rightAnswers / testSize) << "\n";
    std::cout << "Total Error: " << std::to_string(network.mTotalError / testSize) << "\n";
}

void DrawNumber(const std::vector<std::vector<double>>& imageArray, const int label, const int numberIndex, const int networkGuess)
{
    int counter = 0;
    for (int i = 0; i < NumOfPixels; i++)
    {
        if (i % 28 == 0)
            std::cout << "\n";
        if (imageArray[numberIndex][i] > 0)
        {
            std::cout << "*";
        }
        else
        {
            std::cout << " ";
        }
    }
    std::string ans = "\n\tAnswers: " + std::to_string(label);
    std::cout << ans << std::endl;
    std::cout << "Neural Network Answer: " + std::to_string(networkGuess) << "\n";
    std::cout << "Press [q] to quit or any other key to continuen\n";
}

int ReverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void ReadMNIST(std::string path,std::string labelsPath, std::vector<std::vector<double>>& arr, std::vector<int>& labelsArray)
{
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    //Image Reader
    std::ifstream file(path, std::ios::binary);
    if (file.is_open())
    {
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        for (int i = 0; i < number_of_images; ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    arr[i][(n_rows * r) + c] = (double)temp;
                }
            }
        }
    }
    //Label Reader
    std::ifstream file2(labelsPath, std::ios::binary);
    if (file2.is_open())
    {
        file2.read((char*)&magic_number, sizeof(magic_number));  
        magic_number = ReverseInt(magic_number);
        file2.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        for (int i = 0; i < number_of_images; i++)
        {
            char v;
            file2.read(&v, 1);
            labelsArray[i] = (int)v;
        }
    }
}