#include "../Include/pch.h"
#include "../Include/ApplicationManager.h"
#include "../include/DataConstants.h"

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
		{
			averageAccuracy = 0.0;
			averageLoss = 0.0;
			displayManager.epochResults.clear();
			RunNetwork(dataManager.testingData, false);
		}
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
		int totalNumBatches = DataConstants::NUM_TRAINING_IMAGES / hyperParameters.batchSize;
		if (currentBatch >= totalNumBatches)
		{
			displayManager.SaveResults(
				currentEpoch,
				hyperParameters.numEpochs,
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

void ApplicationManager::RunNetwork(const std::vector<std::pair<std::vector<double>, int>>& imageData, bool isTraining)
{
	char n = ' ';
	int counter = 0;
	int rightAnswers = 0;

	neuralNetwork.ClearResults();

	int totalBatchSize = (isTraining) ? hyperParameters.batchSize : DataConstants::NUM_TESTING_IMAGES;

	while (counter < totalBatchSize)
	{
		int val = (isTraining) ? (counter + (totalBatchSize * currentBatch)) : counter;
		int correctAns = imageData[val].second;
		int networkGuess = neuralNetwork.RunNetwork(imageData[val].first);
		if (networkGuess == correctAns)
			rightAnswers++;
		neuralNetwork.CalculateLayerDeltaCost(correctAns);
		counter++;
	}
	averageLoss =	((averageLoss * (currentBatch) + neuralNetwork.mTotalLoss) / (currentBatch + 1));
	averageAccuracy = ((averageAccuracy   * (currentBatch) + ((double)rightAnswers / (double)totalBatchSize)) / (currentBatch + 1));
	
	if (isTraining)
		neuralNetwork.UpdateResults(totalBatchSize);

	displayManager.ClearConsole();
	displayManager.DisplayResults(displayManager.ParseResults(
		currentEpoch,
		hyperParameters.numEpochs,
		currentBatch,
		(DataConstants::NUM_TRAINING_IMAGES / hyperParameters.batchSize),
		averageLoss,
		averageAccuracy
	));
}