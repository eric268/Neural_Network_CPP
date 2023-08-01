#pragma once
#include "pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/DataManager.h"
#include "../Include/DisplayManager.h"

class NeuralNetwork;
class DataManager;
class DisplayManager;

class ApplicationManager
{
public:
	ApplicationManager(NeuralNetwork network, int numEpochs, int batchSize, float learningRate);
	void Run();
	void SaveWeightsAndBias();
	void FitModel();
	void RunNetwork(const std::vector<std::pair<std::vector<float>, int>>& imageData, bool isTraining);

private:
	NeuralNetwork neuralNetwork;
	DataManager dataManager;
	DisplayManager displayManager;
	int numEpochs;
	int batchSize;
	int currentEpoch;
	int currentBatch;
	float learningRate;
};

