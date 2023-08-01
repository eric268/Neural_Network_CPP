#include "../Include/pch.h"
#include "../Include/DisplayManager.h"
#include "../Include/DataConstants.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/DataManager.h"

DisplayManager::DisplayManager() : numImagesToDisplay(25)
{

}

void DisplayManager::DrawNetworkPredictions(NeuralNetwork& network, DataManager& dataManager)
{
	system("CLS");
	for (int i = 0; i < numImagesToDisplay; i++)
	{
		int prediction = network.RunNetwork(dataManager.testingData[i].first);
		DrawNumber(dataManager.testingData[i], prediction);
	}
}

void DisplayManager::DrawNumber(const std::pair<std::vector<float>,int>& imageData, const int networkPrediction)
{
	for (int i = 0; i < DataConstants::NUM_OF_PIXELS_PER_IMAGE; i++)
	{
		if (i % 28 == 0)
			std::cout << "\n";
		if (imageData.first[i] > 0)
			std::cout << "*";
		else
			std::cout << " ";
	}
	std::cout << "\n\tCorrect Answers: " + std::to_string(imageData.second) + '\n';
	std::cout << "Predicted Answer: " + std::to_string(networkPrediction) << "\n";
}

std::string DisplayManager::UserInputOnTrainingCompleted()
{
	std::cout << "\n---------------------------------------------------------\n";
	std::cout << "Enter [train] to continue training:\n";
	std::cout << "Enter [test] to run test images:\n";
	std::cout << "Enter [save] to save weights:\n";
	std::cout << "Enter [display] to display predictions:\n";
	std::cout << "Enter [quit] to exit program\n\n";
	std::string input;
	std::getline(std::cin, input);
	system("CLS");
	return input;
}