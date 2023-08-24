#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/ApplicationManager.h"
#include "../Include/HyperParameters.h"
#include "../Include/DisplayManager.h"
#include "../Include/ActivationFuncTypes.h"

int main()
{
	std::vector<int> networkLayerSizes = std::vector<int>{ 784, 550, 225, 100, 10 };
	NeuralNetwork neuralNetwork (networkLayerSizes, ActivationFunctionTypes::ReLu);
	HyperParameters hyperParameters (64, 1, 0.01);
	ApplicationManager applicationManager (neuralNetwork, hyperParameters);
	applicationManager.Run();
	return 0;
}