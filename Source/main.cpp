#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/ApplicationManager.h"
#include "../Include/HyperParameters.h"
#include "../Include/DisplayManager.h"
#include "../Include/ActivationFuncType.h"

int main()
{
	std::vector<int> networkLayerSizes = std::vector<int>{ 784, 16, 16, 10 };
	NeuralNetwork neuralNetwork ( networkLayerSizes, ActivationFuncType::Sigmoid);
	HyperParameters hyperParameters (0.05, 3, 128);
	ApplicationManager applicationManager (neuralNetwork, hyperParameters);

	applicationManager.Run();
	
	return 0;
}