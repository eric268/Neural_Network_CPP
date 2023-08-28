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
	~NeuralNetwork() = default;

	int  RunNetwork(const std::vector<double> pixelValues);
	void UpdateResults(int testSize);
	void ClearResults();
	void StartBackProp(int correctAns);
	void CalculateLoss(const int correctAns);

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
	void BindActivationFunctions(ActivationFunctionTypes type);
	void PopulateNeuronsInLayers(NetworkLayer* currentLayer);

	std::vector<std::vector<double>> InitalizeNetworkWeights(const int inputSize, const int outputSize);
	std::vector<double> InitalizeBias(const std::size_t layerSize);

	void SetNetworkInputs(std::vector<double> pixelValues);
	void SetHiddenLayersActivation(NetworkLayer* currentLayer);
	void SetOutputLayerActivation(NetworkLayer* outputLayer);
	int  GetFinalOutput(NetworkLayer* outputLayer);

	void CalculateOutputLayerBackProp(NetworkLayer* currentLayer, LayerResults* resultLayer, int correctAns);
	void CalculateHiddenLayerBackProp(NetworkLayer* currentLayer, LayerResults* resultLayer);
	double GetOutputLoss(const int i, const int correctAns, const double outputActivation);


	void ClipGradients(std::vector<std::vector<double>>& weights, double threshold);

	long double totalLoss;
	double learningRate;
	double batchScale;

	std::vector<std::shared_ptr<NetworkLayer>> networkLayers;
	std::vector<std::shared_ptr<LayerResults>> mLayerResults;

	typedef double (*ActivationFuncDelegate)(const double);
	ActivationFuncDelegate ActivationFunction;
	ActivationFuncDelegate D_ActivationFunction;
	ActivationFunctionTypes activationFunctionType;


};
