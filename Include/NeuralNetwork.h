#pragma once
#include "pch.h"
#include "ActivationFunctions.h"
#include "LayerResults.h"

#define InputLayerSize 784
#define HiddenLayer1Size 16
#define HiddenLayer2Size 16
#define OutputLayerSize 10

class NetworkLayer;
class Neurons;
class ActivationFunctions;
enum ActivationFuncType;

class NeuralNetwork
{
public:
	NeuralNetwork() = default;
	NeuralNetwork(std::vector<int>& layerSizes, ActivationFuncType type);

	void ClearResults();
	void LoadWeights(std::string weightPath);
	void BindActivationFunctions(ActivationFuncType type);

	int RunNetwork(const std::vector<double> pixelValues);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer);
	void SetNextLayersActivation(NetworkLayer* currentLayer);
	int GetFinalOutput(NetworkLayer* outputLayer);
	void CalculateLayerDeltaCost(int correctAns);
	void CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void UpdateResults(int testSize);

	double mTotalLoss;
	double learningRate;
	double batchScale;

private:
	std::vector<std::shared_ptr<NetworkLayer>> mNetworkLayers;
	std::vector<std::shared_ptr<LayerResults>> mLayerResults;

	typedef double (*ActivationFuncDelegate)(const double);
	ActivationFuncDelegate ActivationFunction;
	ActivationFuncDelegate D_ActivationFunction;
};
