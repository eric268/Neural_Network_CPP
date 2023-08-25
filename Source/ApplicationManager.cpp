#include "../Include/pch.h"
#include "../Include/ApplicationManager.h"
#include "../include/DataConstants.h"
#include "../Include/Stopwatch.h"

ApplicationManager::ApplicationManager(NeuralNetwork& network, HyperParameters& hyperParameters) :
	neuralNetwork(network),
	hyperParameters(std::move(hyperParameters)),
	dataManager(),
	displayManager(),
	currentEpoch(0),
	currentBatch(0),
	averageLoss(0.0),
	averageAccuracy(0.0)
{
	neuralNetwork.SetLearningRate(hyperParameters.GetLearningRate());
}
void ApplicationManager::Run()
{
	while (true)
	{
		std::string userInput;
		userInput = displayManager.UserInputOnTrainingCompleted();
		if (userInput == "train")
			TrainModel();
		else if (userInput == "test")
			TestModel();
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

void ApplicationManager::SaveNetwork()
{
	std::string fileName;
	std::cout << "Enter save filename: ";
	std::getline(std::cin, fileName);

	if (CheckIfValidFilename(fileName))
		neuralNetwork.SaveWeightsAndBias(fileName);
	else
	{
		DisplayManager::ClearConsole();
		std::cout << "Filename found with invalid characters\n";
	}
}

void ApplicationManager::LoadNetwork()
{
	std::string fileName;
	std::cout << "Enter filename: ";
	std::getline(std::cin, fileName);

	if (std::filesystem::exists("Weights/" + fileName))
		neuralNetwork.LoadWeightsAndBias(fileName);
	else
	{
		DisplayManager::ClearConsole();
		std::cout << "Filename not found\n";
	}
}

void ApplicationManager::FitModel()
{
	currentEpoch = 0;
	currentBatch = 0;
	const int totalNumBatches = DataConstants::NUM_TRAINING_IMAGES / hyperParameters.GetBatchSize();
	neuralNetwork.SetBatchScale(1.0f / hyperParameters.GetBatchSize());
	while (currentEpoch < hyperParameters.GetNumEpochs())
	{
		RunNetwork(dataManager.GetTrainingData(), true);
		currentBatch++;

		if (currentBatch >= totalNumBatches)
		{
			displayManager.SaveResults(
				currentEpoch,
				hyperParameters.GetNumEpochs(),
				totalNumBatches,
				totalNumBatches,
				averageLoss,
				averageAccuracy);

			currentEpoch++;
			currentBatch = 0;
			averageAccuracy = 0.0;
			averageLoss = 0.0;
			dataManager.ShuffleTrainingData();
		}
	}
}

void ApplicationManager::TrainModel()
{
	{
		Stopwatch stopwatch;
		FitModel();
	}
}

void ApplicationManager::TestModel()
{
	averageAccuracy = 0.0;
	averageLoss = 0.0;
	currentBatch = 0;
	currentEpoch = 0;
	displayManager.ClearResults();
	neuralNetwork.SetBatchScale(1.0f / DataConstants::NUM_TESTING_IMAGES);
	std::cout << "Testing...\n";
	RunNetwork(dataManager.GetTestingData(), false);
}

void ApplicationManager::DisplayPredictions()
{
	if (displayManager.DrawPredictionsMenu())
		displayManager.DrawNetworkPredictions(neuralNetwork, dataManager);
}

void ApplicationManager::RunNetwork(const std::vector<std::pair<std::vector<double>, int>>& imageData, bool isTraining)
{
	int counter = 0;
	int rightAnswers = 0;

	neuralNetwork.ClearResults();

	int totalBatchSize = (isTraining) ? hyperParameters.GetBatchSize() : DataConstants::NUM_TESTING_IMAGES;
	int totalBatches   = (isTraining) ? DataConstants::NUM_TRAINING_IMAGES / totalBatchSize : 1;

	while (counter < totalBatchSize)
	{
		int val = (isTraining) ? (counter + (totalBatchSize * currentBatch)) : counter;
		int correctAns = imageData[val].second;
		int networkGuess = neuralNetwork.RunNetwork(imageData[val].first);
		if (networkGuess == correctAns)
			rightAnswers++;

		// Eventually can remove this from testing, but will just need a function at the end that calculates the cross entropy loss from the testing functino
		//if (isTraining)
		{
			neuralNetwork.CalculateLayerDeltaCost(correctAns);
		}

		counter++;
	}

	averageLoss = ((averageLoss * (currentBatch)+neuralNetwork.GetTotalLoss()) / (currentBatch + 1));
	averageAccuracy = ((averageAccuracy * (currentBatch)+((double)rightAnswers / (double)totalBatchSize)) / (currentBatch + 1));
	
	if (isTraining)
	{
		neuralNetwork.UpdateResults(totalBatchSize);
	}

	DisplayManager::ClearConsole();
	displayManager.DisplayResults(displayManager.ParseResults(
		currentEpoch,
		hyperParameters.GetNumEpochs(),
		(currentBatch + 1),
		(totalBatches),
		averageLoss,
		averageAccuracy
	));
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
