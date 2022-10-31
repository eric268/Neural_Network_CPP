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



	int RunOneNumber(const std::vector<double> pixelValues, int answer);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer, NetworkLayer* nextLayer);
	void SetNextLayersActivation(NetworkLayer* currentLayer, NetworkLayer* nextLayer);
	int GetFinalOutput(NetworkLayer* outputLayer);
	void CalculateLayerDeltaCost(int correctAns);
	LayerResults CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, int correctAns);
	LayerResults CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, int correctAns);

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

	double mTotalError;
};

