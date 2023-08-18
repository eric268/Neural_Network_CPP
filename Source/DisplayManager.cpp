#include "../Include/pch.h"
#include "../Include/DisplayManager.h"
#include "../Include/DataConstants.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/DataManager.h"
#include "../Include/ActivationFuncTypes.h"
#include <Windows.h>

DisplayManager::DisplayManager() : numImagesToDisplay(10)
{

}

void DisplayManager::DrawNetworkPredictions(NeuralNetwork& network, DataManager& dataManager, const bool drawIncorrectPredictions)
{
	ClearConsole();

	std::mt19937 rng(std::random_device{}());
	std::uniform_int_distribution<int> distribution(0, DataConstants::NUM_TESTING_IMAGES - 1);

	int imagesToDraw = (drawIncorrectPredictions) ? DataConstants::NUM_TESTING_IMAGES : numImagesToDisplay;
	int counter = 0;

	for (int i = 0; i < imagesToDraw; i++)
	{
		int rand = distribution(rng);
		int num = (drawIncorrectPredictions) ? i : rand;
		int prediction = network.RunNetwork(dataManager.testingData[num].first);

		if (!drawIncorrectPredictions || (drawIncorrectPredictions && prediction != dataManager.testingData[num].second))
		{
			DrawNumber(dataManager.testingData[num], prediction);
			counter++;
			if (counter >= 10)
				break;
		}
	}
}

int DisplayManager::DrawPredictionsMenu()
{
	std::string choice;
	while (true)
	{
		std::cout << "Enter [1] to display up to 10 incorrect predictions\n";
		std::cout << "Enter [2] to display 10 random predictions from the testing dataset\n";
		std::cout << "Enter [back] to return to menu\n";
		std::getline(std::cin, choice);

		if (choice == "1")
		{
			return 1;
		}
		else if (choice == "2")
		{
			return 2;
		}
		else if (choice == "back")
			return -1;
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
	std::cout << "Enter [save] to save model:\n";
	std::cout << "Enter [load] to load model:\n";
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


int DisplayManager::GetNumEpochs()
{
	int numEpochs = 0;
	std::string selection;
	while (true)
	{
		std::cout << "Enter [back] to return to menu\n\n";
		std::cout << "Enter the number of desired epochs\n";
		std::getline(std::cin, selection);

		try
		{
			if (selection == "back")
				return -1;
			else if (std::stoi(selection) <= 0)
			{
				ClearConsole();
				std::cout << "Invalid Input\n";
			}
			else
				return stoi(selection);
		}
		catch (const std::invalid_argument& e)
		{
			ClearConsole();
			std::cout << "Invalid Input\n";
		}
		catch (const std::out_of_range& e)
		{
			std::cout << "Input outside of valid range\n";
		}
	}
}
double DisplayManager::GetLearningRate()
{
	ClearConsole();
	double learningRate = 0.0;
	std::string selection;
	while (true)
	{
		std::cout << "Enter [back] to return to menu\n\n";
		std::cout << "Enter a value between 0 - 1 representing the desired learning rate\n";

		try
		{
			std::getline(std::cin, selection);
			if (selection == "back")
				return -1;
			else if ((std::stod(selection) <= 0 && (std::stod(selection) > 1)))
			{
				ClearConsole();
				std::cout << "Invalid Input\n";
			}
			else
				return stod(selection);
		}
		catch (const std::invalid_argument& e)
		{
			ClearConsole();
			std::cout << "Invalid Input\n";
		}
		catch (const std::out_of_range& e)
		{
			std::cout << "Input outside of valid range\n";
		}

	}
}
int DisplayManager::GetActivationFunction()
{
	ClearConsole();
	int activationFunction = 0;
	std::string selection;
	while (true)
	{
		std::cout << "Enter [back] to return to menu\n\n";
		std::cout << "Enter [1] to use the sigmoid activation function\n";
		std::cout << "Enter [2] to use the ReLu activation function\n";
		std::cout << "Enter [3] to use the Leaky ReLu activation function\n";

		try
		{
			std::getline(std::cin, selection);
			if (selection == "back")
				return -1;
			else if (std::stoi(selection) <= 0 || (std::stoi(selection) > ActivationFunctionTypes::NUM_ACTIVATION_FUNCTIONS))
			{
				ClearConsole();
				std::cout << "Invalid Input\n";
			}
			else
				return stoi(selection);
		}
		catch (const std::invalid_argument& e)
		{
			ClearConsole();
			std::cout << "Invalid Input\n";
		}
		catch (const std::out_of_range& e)
		{
			std::cout << "Input outside of valid range\n";
		}
	}
}
