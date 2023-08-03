#pragma once
#include "pch.h"
#include "MathHelper.h"
#include "LayerResults.h"

#define InputLayerSize 784
#define HiddenLayer1Size 16
#define HiddenLayer2Size 16
#define OutputLayerSize 10

class NetworkLayer;
class Neurons;

class NeuralNetwork
{
public:
	NeuralNetwork();
	NeuralNetwork(std::vector<int>& layerSizes);

	void TrainNetwork();
	void TestNetwork();
	void ClearResults();

	int RunNetwork(const std::vector<double> pixelValues);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer);
	void SetNextLayersActivation(NetworkLayer* currentLayer);
	int GetFinalOutput(NetworkLayer* outputLayer);
	void CalculateLayerDeltaCost(int correctAns);
	void CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void UpdateResults(int testSize);

public:
	std::vector<std::shared_ptr<NetworkLayer>> mNetworkLayers;
	std::vector<std::shared_ptr<LayerResults>> mLayerResults;

	double mTotalError;
	double learningRate;
	double batchScale;
};
