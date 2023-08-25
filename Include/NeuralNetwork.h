#pragma once

#include "LayerResults.h"

class NetworkLayer;
class Neurons;
class ActivationFunctions;
enum ActivationFunctionTypes;

class NeuralNetwork
{
public:
	NeuralNetwork() = default;
	NeuralNetwork(std::vector<int>& layerSizes, ActivationFunctionTypes type);

	void ClearResults();
	void BindActivationFunctions(ActivationFunctionTypes type);

	int  RunNetwork(const std::vector<double> pixelValues);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer);
	void SetNetworkInputs(std::vector<double> pixelValues);
	void SetHiddenLayersActivation(NetworkLayer* currentLayer);
	void SetOutputLayerActivation(NetworkLayer* outputLayer);
	int  GetFinalOutput(NetworkLayer* outputLayer);
	void CalculateLayerDeltaCost(int correctAns);
	void CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void CalculateLayerBackwardsPropagation(NetworkLayer* currentLayer, LayerResults* resultLayer);
	void UpdateResults(int testSize);

	void SaveWeightsAndBias(const std::string& filename) const;
	void LoadWeightsAndBias(const std::string& filename) const;

#pragma region 	Inline Getters& Setters
	const long double GetTotalLoss() const			{ return totalLoss; }
	const double GetLearningRate()   const			{ return learningRate; }
	const double GetBatchScale()     const			{ return batchScale; }
	
	void SetTotalLoss   (const long double loss)	{ totalLoss = loss; }
	void SetLearningRate(const double rate)			{ learningRate = rate; }
	void SetBatchScale  (const double scale)		{ batchScale = scale; }
#pragma endregion

private:
	long double totalLoss;
	double learningRate;
	double batchScale;

	std::vector<std::shared_ptr<NetworkLayer>> networkLayers;
	std::vector<std::shared_ptr<LayerResults>> mLayerResults;

	typedef double (*ActivationFuncDelegate)(const double);
	ActivationFuncDelegate ActivationFunction;
	ActivationFuncDelegate D_ActivationFunction;
	ActivationFunctionTypes activationFunctionType;

	std::vector<std::vector<double>> InitalizeNetworkWeights(const int inputSize, const int outputSize);
	std::vector<double> InitalizeBias(const std::size_t layerSize);
	void ClipGradients(std::vector<std::vector<double>>& weights, double threshold);
};
