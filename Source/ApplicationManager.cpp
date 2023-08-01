#include "../Include/pch.h"
#include "../Include/ApplicationManager.h"
#include "../include/DataConstants.h"

ApplicationManager::ApplicationManager(NeuralNetwork network, int numEpochs, int batchSize, float learningRate) : 
	neuralNetwork {network},
	numEpochs(numEpochs),
	batchSize(batchSize),
	learningRate(learningRate),
	dataManager(),
	displayManager(),
	currentEpoch(0),
	currentBatch(0)
{
	neuralNetwork.learningRate = learningRate;
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
	while (currentEpoch < numEpochs)
	{
		RunNetwork(dataManager.trainingData, true);
		currentBatch++;
		if (currentBatch >= DataConstants::NUM_TRAINING_IMAGES / batchSize)
		{
			currentBatch = 0;
			currentEpoch++;
			dataManager.ShuffleTrainingData();
		}
	}
}
void ApplicationManager::RunNetwork(const std::vector<std::pair<std::vector<float>, int>>& imageData, bool isTraining)
{
	char n = ' ';
	int counter = 0;
	int rightAnswers = 0;

	neuralNetwork.ClearResults();

	while (counter < batchSize)
	{
		int val = (isTraining) ? (counter + (batchSize * currentBatch)) : counter;
		int correctAns = imageData[val].second;
		int networkGuess = neuralNetwork.RunNetwork(imageData[val].first);
		if (networkGuess == correctAns)
			rightAnswers++;
		neuralNetwork.CalculateLayerDeltaCost(correctAns);
		counter++;
	}

	neuralNetwork.UpdateResults(batchSize);
	std::cout << "Correct %" << std::to_string((float)rightAnswers / (float)batchSize) << "\n";
	std::cout << "Total Error: " << std::to_string(neuralNetwork.mTotalError / batchSize) << "\n";
	std::cout << "Epoch: " << std::to_string(currentEpoch) << "/" << std::to_string(numEpochs)
		<< " Batch: " << std::to_string(currentBatch) << "/" << std::to_string((DataConstants::NUM_TRAINING_IMAGES / batchSize) - 1);
}