#pragma once
#include "pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/DataManager.h"
#include "../Include/DisplayManager.h"
#include "../Include/HyperParameters.h"

class ApplicationManager
{
public:
	ApplicationManager(NeuralNetwork& network, HyperParameters& hyperParameters);
	void Run();
	void SaveNetwork();
	void FitModel();
	void RunNetwork(const std::vector<std::pair<std::vector<double>, int>>& imageData, bool isTraining);
	void LoadNetwork();
private:

	bool CheckIfValidFilename(const std::string& filename);
	NeuralNetwork neuralNetwork;
	DataManager dataManager;
	DisplayManager  displayManager;
	HyperParameters hyperParameters;
	int currentEpoch;
	int currentBatch;

	long double averageAccuracy;
	long double averageLoss;
};

