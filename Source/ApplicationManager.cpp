#include "../Include/pch.h"
#include "../Include/ApplicationManager.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/DataManager.h"
#include "../Include/DisplayManager.h"
#include "../Include/HyperParameters.h"
#include "../include/DataConstants.h"
#include "../Include/Stopwatch.h"

ApplicationManager::ApplicationManager(std::unique_ptr<NeuralNetwork> network, std::unique_ptr<HyperParameters> parameters) :
	neuralNetwork	(std::move(network)),
	hyperParameters (std::move(parameters)),
	dataManager		(std::make_unique<DataManager>()),
	displayManager	(std::make_unique<DisplayManager>()),
	currentEpoch	(0),
	currentBatch	(0),
	averageLoss		(0.0),
	averageAccuracy	(0.0)
{
	neuralNetwork->SetLearningRate(hyperParameters->GetLearningRate());
}

ApplicationManager::~ApplicationManager() {}

// Function that handles the basic menu which receives input from the user to decide which action to begin
void ApplicationManager::Run()
{
	while (true)
	{
		displayManager->DisplayMainMenu();
		std::string userInput = GetMenuInput();

		if (userInput == "train")
			StartModelTraining();
		else if (userInput == "test")
			StartModelTest();
		else if (userInput == "save")
			SaveNetwork();
		else if (userInput == "load")
			LoadNetwork();
		else if (userInput == "display")
			DisplayPredictions();
		else if (userInput == "quit")
			break;
	}
}

std::string ApplicationManager::GetMenuInput()
{
	std::string userInput;
	std::getline(std::cin, userInput);
	displayManager->ClearConsole();
	return userInput;
}

void ApplicationManager::StartModelTraining()
{
	displayManager->ClearConsole();
	currentEpoch = 0;
	currentBatch = 0;
	const int totalNumBatches = DataConstants::NUM_TRAINING_IMAGES / hyperParameters->GetBatchSize();
	neuralNetwork->SetBatchScale(1.0f / hyperParameters->GetBatchSize());

	while (currentEpoch < hyperParameters->GetNumEpochs())
	{
		TrainNetwork(dataManager->GetTrainingData());
		currentBatch++;

		// Check if epoch has been completed. If so reset variables and increment completed epochs counter
		if (currentBatch >= totalNumBatches)
		{
			displayManager->SaveResults(
				currentEpoch,
				hyperParameters->GetNumEpochs(),
				totalNumBatches,
				totalNumBatches,
				averageLoss,
				averageAccuracy);

			currentEpoch++;
			currentBatch = 0;
			averageAccuracy = 0.0;
			averageLoss = 0.0;
			dataManager->ShuffleTrainingData();
		}
	}
}

void ApplicationManager::StartModelTest()
{
	displayManager->ClearResults();
	std::cout << "Testing...\n";
	TestNetwork(dataManager->GetTestingData());
}

// Completes one batch of training 
void ApplicationManager::TrainNetwork(const std::vector<std::pair<std::vector<double>, int>>& imageData)
{
	int counter = 0;
	int rightAnswers = 0;
	neuralNetwork->ClearResults();

	while (counter < hyperParameters->GetBatchSize())
	{
		int val = counter + (hyperParameters->GetBatchSize() * currentBatch);
		int correctAns = imageData[val].second;

		// Gets the network prediction
		int networkGuess = neuralNetwork->RunNetwork(imageData[val].first);
		if (networkGuess == correctAns)
			rightAnswers++;

		// Calculates backprop for the output and hidden layers
		neuralNetwork->StartBackProp(correctAns);
		counter++;
	}

	// Updates the network weights and bias's with the loss gradient 
	neuralNetwork->UpdateResults(hyperParameters->GetBatchSize());

	// Calculates a rolling average of the loss and accuraccy for the current epoch
	averageLoss = ((averageLoss * currentBatch) + (neuralNetwork->GetTotalLoss() * neuralNetwork->GetBatchScale())) / (currentBatch + 1);
	averageAccuracy = ((averageAccuracy * currentBatch) + ((double)rightAnswers / (double)hyperParameters->GetBatchSize())) / (currentBatch + 1);

	DisplayManager::ClearConsole();
	displayManager->DisplayResults(displayManager->ParseResults(
		currentEpoch,
		hyperParameters->GetNumEpochs(),
		(currentBatch + 1),
		(DataConstants::NUM_TRAINING_IMAGES / hyperParameters->GetBatchSize()),
		averageLoss,
		averageAccuracy
	));
}

void ApplicationManager::TestNetwork(const std::vector<std::pair<std::vector<double>, int>>& imageData)
{
	int counter = 0;
	int rightAnswers = 0;
	neuralNetwork->SetTotalLoss(0.0);
	neuralNetwork->ClearResults();

	// Iterate through all of the testing data, no batches or epochs
	while (counter < DataConstants::NUM_TESTING_IMAGES)
	{
		int correctAns = imageData[counter].second;
		int networkGuess = neuralNetwork->RunNetwork(imageData[counter].first);
		if (networkGuess == correctAns)
			rightAnswers++;

		// Calculates just the loss of the network, no back prop
		neuralNetwork->CalculateLoss(correctAns);
		counter++;
	}
	const double loss = neuralNetwork->GetTotalLoss() / static_cast<double>(DataConstants::NUM_TESTING_IMAGES);
	const double accuraccy = static_cast<double>(rightAnswers) / static_cast<double>(DataConstants::NUM_TESTING_IMAGES);

	DisplayManager::ClearConsole();
	displayManager->DisplayResults(displayManager->ParseResults(0, 1, 1, 1, loss, accuraccy));
}

void ApplicationManager::DisplayPredictions()
{
	if (displayManager->DrawPredictionsMenu())
		displayManager->DrawNetworkPredictions(neuralNetwork.get(), dataManager.get());
}

void ApplicationManager::LoadNetwork()
{
	std::string fileName;
	std::cout << "Enter filename: ";
	std::getline(std::cin, fileName);

	if (std::filesystem::exists("Weights/" + fileName))
		neuralNetwork->LoadWeightsAndBias(fileName);
	else
	{
		DisplayManager::ClearConsole();
		std::cout << "Filename not found\n";
	}
}

void ApplicationManager::SaveNetwork()
{
	std::string fileName;
	std::cout << "Enter save filename: ";
	std::getline(std::cin, fileName);

	// Ensure filename is valid for windows os
	if (CheckIfValidFilename(fileName))
		neuralNetwork->SaveWeightsAndBias(fileName);
	else
	{
		DisplayManager::ClearConsole();
		std::cout << "Filename found with invalid characters\n";
	}
}

bool ApplicationManager::CheckIfValidFilename(const std::string& filename)
{
	const std::string forbiddenChars = "\\/:*?\"<>|";

	for (char c : filename) {
		if (forbiddenChars.find(c) != std::string::npos) {
			return false;
		}
	}
	return true;
}