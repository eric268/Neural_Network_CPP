#include "../Include/pch.h"
#include "../Include/DisplayManager.h"
#include "../Include/DataConstants.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/DataManager.h"
#include "../Include/ActivationFuncTypes.h"
#include "../Include/Stopwatch.h"
#include <Windows.h>

DisplayManager::DisplayManager() : 
	numImagesToDisplay(0)
{}

void DisplayManager::DrawNetworkPredictions(NeuralNetwork* network, DataManager* dataManager)
{
	ClearConsole();

	std::mt19937 rng(std::random_device{}());
	std::uniform_int_distribution<int> distribution(0, DataConstants::NUM_TESTING_IMAGES - 1);

	for (int i = 0; i < numImagesToDisplay; i++)
	{
		int rand = distribution(rng);
		int prediction = network->RunNetwork(dataManager->GetTestingData()[rand].first);
		std::cout << GetNumberDisplay(dataManager->GetTestingData()[rand], prediction);
	}
	
}

bool DisplayManager::DrawPredictionsMenu()
{
	std::string input;
	while (true)
	{
		std::cout << "Enter [back] to return to menu\n\n";
		std::cout << "Enter the amount of numbers to display: ";
		std::getline(std::cin, input);

		if (input == "back")
			return false;

		try
		{
			int num = std::stoi(input);
			if (num >= DataConstants::NUM_TESTING_IMAGES)
				throw std::out_of_range("Amount selected greater than total number of images. Ensure the input is less than 10,000\n");
			else
			{
				numImagesToDisplay = num;
				return true;
			}
		}
		catch (std::invalid_argument&)
		{
			std::cerr << "Invalid input" << '\n';
		}
		catch (std::out_of_range& e)
		{
			std::cerr << e.what() << '\n';
		}
	}
}

std::string DisplayManager::GetNumberDisplay(const std::pair<std::vector<double>, int>& imageData, const int networkPrediction)
{
	// Calculate the size of the char array: 784 characters for '*' or ' '
	// plus 27 newlines.
	const size_t arraySize = DataConstants::NUM_OF_PIXELS_PER_IMAGE + 27;
	std::vector<char> output(arraySize);

	try
	{
		size_t index = 0;
		for (int i = 0; i < DataConstants::NUM_OF_PIXELS_PER_IMAGE; i++)
		{
			if (i % 28 == 0)
			{
				output[index++] = '\n';
			}
			if (imageData.first[i] > 0)
			{
				output[index++] = '*';
			}
			else
			{
				output[index++] = ' ';
			}
		}

		// Convert the character array to a std::string before returning.
		std::string displayString = std::string(output.begin(), output.end());
		displayString += "\nCorrect Answers: " + std::to_string(imageData.second) + '\n';
		displayString += "Predicted Answer: " + std::to_string(networkPrediction) + "\n";
		return displayString;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw;
	}
}

void DisplayManager::DisplayMainMenu()
{
	std::cout << "---------------------------------------------------------\n";
	std::cout << "Enter [train] to continue training\n";
	std::cout << "Enter [test] to run test images\n";
	std::cout << "Enter [save] to save model\n";
	std::cout << "Enter [load] to load model\n";
	std::cout << "Enter [display] to display predictions\n";
	std::cout << "Enter [quit] to exit program\n\n";
}

std::string DisplayManager::ParseResults(const int currentEpoch, const int maxEpoch, const int currentBatch, const int totalBatches, long double loss, long double accuracy)
{
	return "Epoch: " + std::to_string(currentEpoch + 1)  + "/" + std::to_string(maxEpoch) 
		+ "\nBatch: " + std::to_string(currentBatch)   + "/" + std::to_string((totalBatches))
		+ "  -  Loss: " + std::to_string(loss) + "  -  Accuracy " + std::to_string(accuracy);
}

void DisplayManager::DisplayResults(std::string results)
{
	std::string output;

	try
	{
		for (const auto& s : epochResults)
			output += s + '\n';
		output += results + '\n';
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw;
	}

	std::cout << output << '\n';
}

void DisplayManager::ClearConsole()
{
	static const HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	COORD topLeft = { 0, 0 };
	std::cout.flush();

	// Get the current buffer info
	if (!GetConsoleScreenBufferInfo(hOut, &csbi)) {
		abort();
	}

	// Set a new buffer size to allow more lines
	COORD newBufferSize;
	newBufferSize.X = csbi.dwSize.X;  // Keep the current width
	newBufferSize.Y = 3000;           // Increase the height
	if (!SetConsoleScreenBufferSize(hOut, newBufferSize)) {
		// Handle error
		std::cerr << "Could not set the new buffer size.";
		abort();
	}

	// Only clear the visible window, not the entire buffer
	DWORD length = csbi.dwSize.X * (csbi.srWindow.Bottom - csbi.srWindow.Top + 1);
	DWORD written;

	// Clear characters
	FillConsoleOutputCharacter(hOut, TEXT(' '), length, topLeft, &written);

	// Clear attributes
	FillConsoleOutputAttribute(hOut, csbi.wAttributes, length, topLeft, &written);

	// Reset the cursor position
	SetConsoleCursorPosition(hOut, topLeft);
}
