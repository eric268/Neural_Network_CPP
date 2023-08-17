#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/ActivationFuncType.h"
#include "../Include/WeightInitializer.h"

NeuralNetwork::NeuralNetwork(std::vector<int>& layerSizes, ActivationFuncType type) : 
	mTotalLoss{ 0.0 }, 
	activationFunctionType{type}
{
	assert(layerSizes.size() > 0);

	mNetworkLayers = std::vector<std::shared_ptr<NetworkLayer>>(layerSizes.size());
	mNetworkLayers[0] = std::make_shared<NetworkLayer>(layerSizes[0]);
	mLayerResults = std::vector<std::shared_ptr<LayerResults>>(layerSizes.size() - 1);

	for (int i = 1; i < layerSizes.size(); i++)
	{
		mNetworkLayers[i] = std::make_shared<NetworkLayer>(layerSizes[i]);
		mNetworkLayers[i -1]->mNextLayer = mNetworkLayers[i];
		mNetworkLayers[i]->mPreviousLayer = mNetworkLayers[i - 1];
		mLayerResults[i - 1] = std::make_shared<LayerResults>(layerSizes[i], layerSizes[i - 1]);
		PopulateNeuronsInLayers(mNetworkLayers[i].get());
	}

	BindActivationFunctions(activationFunctionType);
}

void NeuralNetwork::PopulateNeuronsInLayers(NetworkLayer* currentLayer)
{
	if (!currentLayer || !currentLayer->mPreviousLayer)
		return;

	const int currentLayerSize = currentLayer->mNumberOfNeurons;
	const int prevLayerSize = currentLayer->mPreviousLayer->mNumberOfNeurons;

	currentLayer->mWeights = std::vector<std::vector<double>>(currentLayerSize, std::vector<double>(prevLayerSize));
	InitalizeNetworkWeights(currentLayer->mWeights, prevLayerSize, currentLayerSize);

	currentLayer->mBias = std::vector<double>(currentLayerSize);
	InitalizeBias(currentLayer->mBias);
}


void NeuralNetwork::ClearResults()
{
	mTotalLoss = 0.0;

	for (int i = 0; i < mLayerResults.size(); i++)
	{
		std::fill(mLayerResults[i]->mBiasResults.begin(), mLayerResults[i]->mBiasResults.end(), 0.0);

		for (auto& p :mLayerResults[i]->mWeightedResults)
		{
			std::fill(p.begin(), p.end(), 0.0);
		}
	}
}

int NeuralNetwork::RunNetwork(std::vector<double> pixelValues)
{
	// Input Layer
	for (int i = 0; i < pixelValues.size(); i++)
		mNetworkLayers[0]->mNeurons[i]->mActivation = pixelValues[i] / 255.0;

	// Hidden Layers
	for (int i = 1; i < mNetworkLayers.size() - 1; i++)
		SetHiddenLayersActivation(mNetworkLayers[i].get());

	//Output Layer
	SetOutputLayerActivation(mNetworkLayers[mNetworkLayers.size() - 1].get());

	return GetFinalOutput(mNetworkLayers[mNetworkLayers.size() - 1].get());
}

void NeuralNetwork::SetHiddenLayersActivation(NetworkLayer* currentLayer)
{
	try
	{
		if (!currentLayer || !currentLayer->mPreviousLayer)
			throw std::runtime_error("Null layer passed in SetHiddenLayersActivation");
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what();
		throw;
	}

	const int currentLayerSize = currentLayer->mNumberOfNeurons;
	const int prevLayerSize = currentLayer->mPreviousLayer->mNumberOfNeurons;

	for (int i = 0; i < currentLayerSize; i++)
	{
		currentLayer->mNeurons[i]->mActivation = 0.0;
		for (int j = 0; j < prevLayerSize; j++)
		{
			currentLayer->mNeurons[i]->mActivation += currentLayer->mWeights[i][j] * currentLayer->mPreviousLayer->mNeurons[j]->mActivation;
		}

		// Ensure that the activation function matches the derivative 
		const float activation = currentLayer->mNeurons[i]->mActivation + currentLayer->mBias[i];
		currentLayer->mNeurons[i]->mActivation = ActivationFunction(activation);
	}
}

void NeuralNetwork::SetOutputLayerActivation(NetworkLayer* outputLayer)
{
	try
	{
		if (!outputLayer)
			throw std::runtime_error("Null layer passed in SetOutputLayerActivation");
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what();
		throw;
	}

	const int currentLayerSize = outputLayer->mNumberOfNeurons;
	const int prevLayerSize = outputLayer->mPreviousLayer->mNumberOfNeurons;

	for (int i = 0; i < currentLayerSize; i++)
	{
		outputLayer->mNeurons[i]->mActivation = 0.0;
		for (int j = 0; j < prevLayerSize; j++)
		{
			outputLayer->mNeurons[i]->mActivation += outputLayer->mWeights[i][j] * outputLayer->mPreviousLayer->mNeurons[j]->mActivation;
		}
		outputLayer->mNeurons[i]->mActivation += outputLayer->mBias[i];
	}

	std::vector<double> logits(outputLayer->mNumberOfNeurons, 0);
	for (int i = 0; i < outputLayer->mNumberOfNeurons; i++)
	{
		logits[i] = outputLayer->mNeurons[i]->mActivation + outputLayer->mBias[i];
	}

	auto probailities = ActivationFunctions::softmax(logits);
	for (int i = 0; i < outputLayer->mNumberOfNeurons; i++)
		outputLayer->mNeurons[i]->mActivation = probailities[i];
}

int NeuralNetwork::GetFinalOutput(NetworkLayer* outputLayer)
{
	double highestActivation = -INFINITY;
	int i, ans = -1;
	for (i = 0; i < outputLayer->mNeurons.size(); i++)
	{
		if (highestActivation < outputLayer->mNeurons[i]->mActivation)
		{
			highestActivation = outputLayer->mNeurons[i]->mActivation;
			ans = i;
		}
	}
	return ans;
}


void NeuralNetwork::CalculateLayerDeltaCost(int correctAns)
{
	//Output layer back prop is different function from other layers so is done separately 
	CalculateOutputLayerBackwardsProp(mNetworkLayers[mNetworkLayers.size() - 1].get(), mLayerResults[mLayerResults.size() - 1].get(), correctAns);

	for (int i = mNetworkLayers.size() - 2; i > 0; i--)
		CalculateLayerBackwardsPropigation(mNetworkLayers[i].get(), mLayerResults[i - 1].get(), correctAns);
}

void NeuralNetwork::CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* layerResults, int correctAns)
{
	double y = 0.0;
	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		y = (correctAns == i) ? 1.0 : 0.0;
		const double activation = currentLayer->mNeurons[i]->mActivation;
		const double deltaError = 2.0 * (activation - y);
		// Ensure that the activation function matches the derivative 
		const double deltaOutput = D_ActivationFunction(activation);

		mTotalLoss += 0.5 * ((activation - y) * (activation - y)) * batchScale;

		currentLayer->mNeurons[i]->mDeltaError = deltaError;
		currentLayer->mNeurons[i]->mDeltaOutput = deltaOutput;
		layerResults->mBiasResults[i] = deltaError * deltaOutput;

		for (int j = 0; j < currentLayer->mPreviousLayer->mNumberOfNeurons; j++)
		{
			const double prevLayerActivation = currentLayer->mPreviousLayer->mNeurons[j]->mActivation;
			layerResults->mWeightedResults[i][j] += (deltaError * deltaOutput * prevLayerActivation) * batchScale;
		}
	}
}

void NeuralNetwork::CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, LayerResults* layerResults, int correctAns)
{
	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		const double activation = currentLayer->mNeurons[i]->mActivation;
		// Ensure that the activation function matches the derivative 
		const double deltaOutput = D_ActivationFunction(activation);
		currentLayer->mNeurons[i]->mDeltaError = 0.0;
		currentLayer->mNeurons[i]->mDeltaOutput = deltaOutput;


		for (int j = 0; j < currentLayer->mNextLayer->mNumberOfNeurons; j++)
		{
			currentLayer->mNeurons[i]->mDeltaError += (
				currentLayer->mNextLayer->mNeurons[j]->mDeltaError		* 
				currentLayer->mNextLayer->mNeurons[j]->mDeltaOutput		* 
				currentLayer->mNextLayer->mWeights[j][i]);
		}

		layerResults->mBiasResults[i] = currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput;
	}

	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->mPreviousLayer->mNumberOfNeurons; j++)
		{
			layerResults->mWeightedResults[i][j] += (
				currentLayer->mNeurons[i]->mDeltaError					* 
				currentLayer->mNeurons[i]->mDeltaOutput					* 
				currentLayer->mPreviousLayer->mNeurons[j]->mActivation);
		}
	}
}

void NeuralNetwork::UpdateResults(int testSize)
{
	for (int i = 1; i < mNetworkLayers.size(); i++)
		mNetworkLayers[i]->UpdateBias(mLayerResults[i - 1].get(), learningRate);

	for (int i = mNetworkLayers.size() - 1; i > 0; i--)
		mNetworkLayers[i]->UpdateWeight(mLayerResults[i - 1].get(), learningRate);
}

void NeuralNetwork::LoadWeights(std::string weightPath)
{
	if (!weightPath.size())
		return;

	// TODO:
	// Create functionality to load weights
}

void NeuralNetwork::BindActivationFunctions(ActivationFuncType type)
{
	switch (type)
	{
	case ActivationFuncType::Sigmoid:
		ActivationFunction = ActivationFunctions::Sigmoid;
		D_ActivationFunction = ActivationFunctions::D_Sigmoid;
		break;
	case ActivationFuncType::ReLU:
		ActivationFunction = ActivationFunctions::ReLU;
		D_ActivationFunction = ActivationFunctions::D_ReLU;
		break;
	case ActivationFuncType::Leaky_ReLU:
		ActivationFunction = ActivationFunctions::LeakyReLU;
		D_ActivationFunction = ActivationFunctions::D_Leaky_ReLU;
		break;
	default:
		throw std::runtime_error("Invalid activation function passed to network");
	}
}

void NeuralNetwork::InitalizeNetworkWeights(std::vector<std::vector<double>>& weights, const int inputSize, const int outputSize)
{
	for (int i = 0; i < outputSize; i++)
	{
		for (int j = 0; j < inputSize; j++)
		{
			switch (activationFunctionType)
			{
			case Sigmoid:
				weights[i][j] = WeightInitializer::Xavier(inputSize, outputSize);
				break;
			case ReLU:
				weights[i][j] = WeightInitializer::He(inputSize);
				break;
			case Leaky_ReLU:
				weights[i][j] = WeightInitializer::He(inputSize);
				break;
			}
		}
	}
}

void NeuralNetwork::InitalizeBias(std::vector<double>& bias)
{
	for (int i =0; i < bias.size(); i++)
	{
		switch (activationFunctionType)
		{
		case Sigmoid:
			bias[i] = 0.0;
			break;
		case ReLU:
			bias[i] = 0.01;
			break;
		case Leaky_ReLU:
			bias[i] = 0.0;
			break;
		}
	}
}