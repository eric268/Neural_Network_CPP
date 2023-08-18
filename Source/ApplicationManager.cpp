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
	std::string saveWeightName;
	std::cout << "Enter save filename: ";
	std::getline(std::cin, saveWeightName);

	if (CheckIfValidFilename(saveWeightName))
		dataManager.SaveWeightsAndBias(neuralNetwork.GetFirstHiddenLayer(), saveWeightName);
	else
	{
		DisplayManager::ClearConsole();
		std::cout << "Filename found with invalid characters\n";
	}
}

void ApplicationManager::LoadNetwork()
{
	std::string loadWeightName;
	std::cout << "Enter filename: ";
	std::getline(std::cin, loadWeightName);

	if (std::filesystem::exists("Weights/" + loadWeightName))
		dataManager.LoadWeightsAndBias(neuralNetwork.GetFirstHiddenLayer(), loadWeightName);
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

void ApplicationManager::TrainModel()
{
	int epochs = displayManager.GetNumEpochs();
	double learningRate = displayManager.GetLearningRate();
	int activationFunction = displayManager.GetActivationFunction();

	if (!epochs || !learningRate || activationFunction < 0)
		return;

	hyperParameters.numEpochs = epochs;
	neuralNetwork.learningRate = learningRate;
	neuralNetwork.BindActivationFunctions(activationFunction);

	FitModel();
}

void ApplicationManager::TestModel()
{
	averageAccuracy = 0.0;
	averageLoss = 0.0;
	displayManager.epochResults.clear();
	std::cout << "Testing...\n";
	RunNetwork(dataManager.testingData, false);
}

void ApplicationManager::DisplayPredictions()
{
	int ans = displayManager.DrawPredictionsMenu();

	switch (ans)
	{
	case -1:
		return;
	case 1:
		displayManager.DrawNetworkPredictions(neuralNetwork, dataManager, true);
		break;
	case 2:
		displayManager.DrawNetworkPredictions(neuralNetwork, dataManager, false);
		break;
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

	DisplayManager::ClearConsole();
	displayManager.DisplayResults(displayManager.ParseResults(
		currentEpoch,
		hyperParameters.numEpochs,
		currentBatch,
		(DataConstants::NUM_TRAINING_IMAGES / hyperParameters.batchSize),
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
