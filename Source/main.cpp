#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/ApplicationManager.h"
#include "../Include/HyperParameters.h"
#include "../Include/DisplayManager.h"
#include "../Include/ActivationFuncTypes.h"

int main()
{
	std::vector<int> networkLayerSizes = std::vector<int>{ 784, 550, 225, 100, 10 };
	std::unique_ptr<NeuralNetwork> neuralNetwork = std::make_unique<NeuralNetwork>(networkLayerSizes, ActivationFunctionTypes::ReLU);
	std::unique_ptr<HyperParameters> hyperParameters = std::make_unique<HyperParameters>(64, 3, 0.05);
	ApplicationManager applicationManager (std::move(neuralNetwork),std::move(hyperParameters));
	applicationManager.Run();
	return 0;
}