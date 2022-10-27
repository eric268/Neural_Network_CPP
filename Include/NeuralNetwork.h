#pragma once
#include "pch.h"
#include "MathHelper.h"
#include "LayerResults.h"


class NetworkLayer;
class Neurons;

class NeuralNetwork
{
public:
	NeuralNetwork();



	int RunOneNumber(NetworkLayer* inputLayer, NetworkLayer* outputLayer, const std::vector<double> pixelValues, int answer);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer, NetworkLayer* nextLayer);
	void SetNextLayersActivation(NetworkLayer* currentLayer, NetworkLayer* nextLayer);
	int GetFinalOutput(NetworkLayer* outputLayer);
	void CalculateOutputLayerCost(int correctAns);
	LayerResults OutputLayerBackwardsInduction(NetworkLayer* currentLayer, NetworkLayer* prevLayer);

public:
	NetworkLayer* mInputLayer;
	NetworkLayer* mHiddenLayer1;
	NetworkLayer* mHiddenLayer2;
	NetworkLayer* mOutputLayer;

	NetworkLayer* mTestInput1;
	NetworkLayer* mTestOutput1;

	LayerResults mHiddenLayer1Results;
	LayerResults mHiddenLayer2Results;
	LayerResults mOutputLayerResults;

	double mCostFound;
	double mDeltaCost;
};

