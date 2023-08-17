#include "../Include/pch.h"
#include "../Include/DisplayManager.h"
#include "../Include/DataConstants.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/DataManager.h"

#include <Windows.h>

DisplayManager::DisplayManager() : numImagesToDisplay(25)
{

}

void DisplayManager::DrawNetworkPredictions(NeuralNetwork& network, DataManager& dataManager)
{
	ClearConsole();
	for (int i = 0; i < numImagesToDisplay; i++)
	{
		int prediction = network.RunNetwork(dataManager.testingData[i].first);
		DrawNumber(dataManager.testingData[i], prediction);
	}
}

void DisplayManager::DrawNumber(const std::pair<std::vector<double>,int>& imageData, const int networkPrediction)
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
	std::cout << "\nCorrect Answers: " + std::to_string(imageData.second) + '\n';
	std::cout << "Predicted Answer: " + std::to_string(networkPrediction) << "\n";
}

std::string DisplayManager::UserInputOnTrainingCompleted()
{
	std::cout << "---------------------------------------------------------\n";
	std::cout << "Enter [train] to continue training:\n";
	std::cout << "Enter [test] to run test images:\n";
	std::cout << "Enter [save] to save weights:\n";
	std::cout << "Enter [display] to display predictions:\n";
	std::cout << "Enter [quit] to exit program\n\n";
	std::string input;
	std::getline(std::cin, input);
	ClearConsole();
	return input;
}

std::string DisplayManager::ParseResults(const int currentEpoch, const int maxEpoch, const int currentBatch, const int totalBatches, long double loss, long double accuracy)
{
	return "Epoch: " + std::to_string(currentEpoch + 1)  + "/" + std::to_string(maxEpoch) 
		+ "\nBatch: " + std::to_string(currentBatch)   + "/" + std::to_string((totalBatches))
		+ "  -  Loss: " + std::to_string(loss) + "  -  Accuracy " + std::to_string(accuracy);
}

void DisplayManager::DisplayResults(std::string results)
{
	for (const auto& s : epochResults)
		std::cout << s << '\n';
	std::cout << results << '\n';
}

void DisplayManager::ClearConsole()
{
	static const HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	COORD topLeft = { 0, 0 };
	std::cout.flush();

	if (!GetConsoleScreenBufferInfo(hOut, &csbi)) {
		abort();
	}
	DWORD length = csbi.dwSize.X * csbi.dwSize.Y;
	DWORD written;
	FillConsoleOutputCharacter(hOut, TEXT(' '), length, topLeft, &written);
	FillConsoleOutputAttribute(hOut, csbi.wAttributes, length, topLeft, &written);
	SetConsoleCursorPosition(hOut, topLeft);
}