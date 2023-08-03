#include "../Include/pch.h"
#include "../Include/ApplicationManager.h"
#include "../include/DataConstants.h"

ApplicationManager::ApplicationManager(NeuralNetwork& network, HyperParameters& hyperParameters) :
	neuralNetwork(network),
	hyperParameters(std::move(hyperParameters)),
	dataManager(),
	displayManager(),
	currentEpoch(0),
	currentBatch(0)
{
	neuralNetwork.learningRate = this->hyperParameters.learningRate;
	neuralNetwork.batchScale = (1.0f / this->hyperParameters.batchSize);
}
void ApplicationManager::Run()
{
	while (true)
	{
		std::string userInput;
		userInput = displayManager.UserInputOnTrainingCompleted();
		if (userInput == "train")
			FitModel();
		else if (userInput == "test")
			RunNetwork(dataManager.testingData, false);
		else if (userInput == "save")
			SaveWeightsAndBias();
		else if (userInput == "display")
			displayManager.DrawNetworkPredictions(neuralNetwork, dataManager);
		else if (userInput == "quit")
			break;
	}
}

void ApplicationManager::SaveWeightsAndBias()
{
	std::string saveWeightName, saveBiasName;
	std::cout << "Enter save weight name: ";
	std::cin >> saveWeightName;
	std::cout << "Enter save bias name: ";
	std::cin >> saveBiasName;
}
void ApplicationManager::FitModel()
{
	currentEpoch = 0;
	currentBatch = 0;
	while (currentEpoch < hyperParameters.numEpochs)
	{
		RunNetwork(dataManager.trainingData, true);
		currentBatch++;
		if (currentBatch >= DataConstants::NUM_TRAINING_IMAGES / hyperParameters.batchSize)
		{
			currentBatch = 0;
			currentEpoch++;
			dataManager.ShuffleTrainingData();
		}
	}
}
void ApplicationManager::RunNetwork(const std::vector<std::pair<std::vector<double>, int>>& imageData, bool isTraining)
{
	char n = ' ';
	int counter = 0;
	int rightAnswers = 0;

	neuralNetwork.ClearResults();

	while (counter < hyperParameters.batchSize)
	{
		int val = (isTraining) ? (counter + (hyperParameters.batchSize * currentBatch)) : counter;
		int correctAns = imageData[val].second;
		int networkGuess = neuralNetwork.RunNetwork(imageData[val].first);
		if (networkGuess == correctAns)
			rightAnswers++;
		neuralNetwork.CalculateLayerDeltaCost(correctAns);
		counter++;
	}

	neuralNetwork.UpdateResults(hyperParameters.batchSize);
	std::cout << "Correct %" << std::to_string((double)rightAnswers / (double)hyperParameters.batchSize) << "\n";
	std::cout << "Total Error: " << std::to_string(neuralNetwork.mTotalError) << "\n";
	std::cout << "Epoch: " << std::to_string(currentEpoch) << "/" << std::to_string(hyperParameters.numEpochs)
		<< " Batch: " << std::to_string(currentBatch) << "/" << std::to_string((DataConstants::NUM_TRAINING_IMAGES / hyperParameters.batchSize) - 1);
}