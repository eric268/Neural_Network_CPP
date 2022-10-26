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

	int RunOneNumber(std::vector<double> pixelValues, int answer);
	void PopulateNeuronsInLayers(NetworkLayer& currentLayer, NetworkLayer& nextLayer);
	void SetNextLayersActivation(NetworkLayer& currentLayer, NetworkLayer& nextLayer);

public:
	NetworkLayer* mInputLayer;
	NetworkLayer* mHiddenLayer1;
	NetworkLayer* mHiddenLayer2;
	NetworkLayer* mOutputLayer;
};

