#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/ApplicationManager.h"
#include "../Include/HyperParameters.h"
#include "../Include/DisplayManager.h"
#include "../Include/ActivationFuncTypes.h"

int main()
{
	std::vector<int> networkLayerSizes = std::vector<int>{ 784, 550, 250, 100, 10 };
	NeuralNetwork neuralNetwork ( networkLayerSizes, ActivationFunctionTypes::LeakyReLu);
	HyperParameters hyperParameters (0.01, 1, 32);
	ApplicationManager applicationManager (neuralNetwork, hyperParameters);

	applicationManager.Run();
	
	return 0;
}