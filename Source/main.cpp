#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/ApplicationManager.h"

int main()
{
	NeuralNetwork neuralNetwork(std::vector<int>{ 784, 16, 16, 10 });
	ApplicationManager applicationManager(neuralNetwork, 10, 128, 0.05f);

	applicationManager.Run();
	
	return 0;
}