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
	NeuralNetwork(std::vector<int> layerSizes);

	int RunNetwork(const std::vector<double> pixelValues);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer);
	void SetNextLayersActivation(NetworkLayer* currentLayer);
	int GetFinalOutput(NetworkLayer* outputLayer);
	void CalculateLayerDeltaCost(int correctAns);
	void CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void UpdateResults(int testSize);

public:
	NetworkLayer* mInputLayer;
	NetworkLayer* mHiddenLayer1;
	NetworkLayer* mHiddenLayer2;
	NetworkLayer* mOutputLayer;

	LayerResults mHiddenLayer1Results;
	LayerResults mHiddenLayer2Results;
	LayerResults mOutputLayerResults;

	std::vector<NetworkLayer*> mNetworkLayers;
	std::vector<LayerResults*> mLayerResults;

	double mTotalError;
};

