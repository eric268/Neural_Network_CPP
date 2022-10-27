#pragma once
#include "pch.h"
#include "MathHelper.h"


class NetworkLayer;
class Neurons;

class NeuralNetwork
{
public:
	NeuralNetwork();

	void SerializeWeights();
	void SerializeBias();

	void BeginTraining();

	void BackwardPropogate();

	void BeingTesting();

	void CalculateCosts(int correctAns);

	int RunOneNumber(NetworkLayer* inputLayer, NetworkLayer* outputLayer, std::vector<double> pixelValues, int answer);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer, NetworkLayer* nextLayer);
	void SetNextLayersActivation(NetworkLayer* currentLayer, NetworkLayer* nextLayer);
	int GetFinalOutput(NetworkLayer* outputLayer);

public:
	NetworkLayer* mInputLayer;
	NetworkLayer* mHiddenLayer1;
	NetworkLayer* mHiddenLayer2;
	NetworkLayer* mOutputLayer;

	NetworkLayer* mTestInput1;
	NetworkLayer* mTestOutput1;
};

