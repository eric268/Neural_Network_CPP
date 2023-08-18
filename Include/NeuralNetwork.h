#pragma once
#include "pch.h"
#include "LayerResults.h"

class NetworkLayer;
class Neurons;
class ActivationFunctions;

class NeuralNetwork
{
public:
	NeuralNetwork() = default;
	NeuralNetwork(std::vector<int>& layerSizes, int type);

	void ClearResults();
	void LoadWeights(std::string weightPath);
	void BindActivationFunctions(int type);

	int RunNetwork(const std::vector<double> pixelValues);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer);
	void SetHiddenLayersActivation(NetworkLayer* currentLayer);
	void SetOutputLayerActivation(NetworkLayer* outputLayer);
	int GetFinalOutput(NetworkLayer* outputLayer);
	void CalculateLayerDeltaCost(int correctAns);
	void CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void UpdateResults(int testSize);

	long double mTotalLoss;
	double learningRate;
	double batchScale;

	inline NetworkLayer* GetFirstHiddenLayer()
	{
		return (mNetworkLayers.size() > 1) ? mNetworkLayers[1].get() : nullptr;
	}

private:
	std::vector<std::shared_ptr<NetworkLayer>> mNetworkLayers;
	std::vector<std::shared_ptr<LayerResults>> mLayerResults;

	typedef double (*ActivationFuncDelegate)(const double);
	ActivationFuncDelegate ActivationFunction;
	ActivationFuncDelegate D_ActivationFunction;
	int activationFunctionType;

	void InitalizeNetworkWeights(std::vector<std::vector<double>>& weights, const int inputSize, const int outputSize);
	void InitalizeBias(std::vector<double>& bias);
};
