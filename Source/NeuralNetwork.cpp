#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/ActivationFuncType.h"

NeuralNetwork::NeuralNetwork(std::vector<int>& layerSizes, ActivationFuncType type) : mTotalLoss{ 0.0 }
{
	assert(layerSizes.size() > 0);

	mNetworkLayers = std::vector<std::shared_ptr<NetworkLayer>>(layerSizes.size());
	mNetworkLayers[0] = std::make_shared<NetworkLayer>(layerSizes[0]);
	mLayerResults = std::vector<std::shared_ptr<LayerResults>>(layerSizes.size() - 1);

	for (int i = 1; i < layerSizes.size(); i++)
	{
		mNetworkLayers[i] = std::make_shared<NetworkLayer>(layerSizes[i]);
		mNetworkLayers[i - 1]->mNextLayer = mNetworkLayers[i];
		mNetworkLayers[i]->mPreviousLayer = mNetworkLayers[i - 1];
		mLayerResults[i - 1] = std::make_shared<LayerResults>(layerSizes[i], layerSizes[i - 1]);
		PopulateNeuronsInLayers(mNetworkLayers[i - 1].get());
	}

	BindActivationFunctions(type);
}

void NeuralNetwork::PopulateNeuronsInLayers(NetworkLayer* currentLayer)
{
	if (!currentLayer || !currentLayer->mNextLayer)
		return;

	currentLayer->mWeights = std::vector<std::vector<double>>(currentLayer->mNextLayer->mNumberOfNeurons, std::vector<double>(currentLayer->mNumberOfNeurons));
	std::random_device rd;
	std::uniform_int_distribution<int> dist(-10000, 10000);

	for (int i = 0; i < currentLayer->mNextLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->mNumberOfNeurons; j++)
		{
			double val = ((double)dist(rd)) * 0.00001;
			currentLayer->mWeights[i][j] = val;
		}
	}
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
	for (int i = 0; i < pixelValues.size(); i++)
		mNetworkLayers[0]->mNeurons[i]->mActivation = pixelValues[i] / 255.0;

	for (int i = 0; i < mNetworkLayers.size() - 1; i++)
		SetNextLayersActivation(mNetworkLayers[i].get());

	return GetFinalOutput(mNetworkLayers[mNetworkLayers.size() - 1].get());
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

void NeuralNetwork::SetNextLayersActivation(NetworkLayer* currentLayer)
{
	std::vector<double> logits;
	for (int i = 0; i < currentLayer->mNextLayer->mNumberOfNeurons; i++)
	{
		currentLayer->mNextLayer->mNeurons[i]->mActivation = 0.0;
		for (int j = 0; j < currentLayer->mNumberOfNeurons; j++)
		{
			currentLayer->mNextLayer->mNeurons[i]->mActivation += currentLayer->mWeights[i][j] * currentLayer->mNeurons[j]->mActivation;
		}
		// Check to make sure that this is not the output layer
		if (currentLayer->mNextLayer->mNextLayer)
		{
			// Ensure that the activation function matches the derivative 
			const float activation = currentLayer->mNextLayer->mNeurons[i]->mActivation + currentLayer->mNextLayer->mNeurons[i]->mBias;
			currentLayer->mNextLayer->mNeurons[i]->mActivation = ActivationFunction(activation);
		}
		else
			// Is output layer so will use softmax not other activation functions
			logits.push_back(currentLayer->mNextLayer->mNeurons[i]->mActivation + currentLayer->mNextLayer->mNeurons[i]->mBias);
	}
	if (logits.size())
	{
		auto probailities = ActivationFunctions::softmax(logits);
		for (int i = 0; i < currentLayer->mNextLayer->mNumberOfNeurons; i++)
			currentLayer->mNextLayer->mNeurons[i]->mActivation = probailities[i];
	}

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

		mTotalLoss += ((activation - y) * (activation - y)) * batchScale;

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
				currentLayer->mWeights[j][i]);
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

	for (int i = mNetworkLayers.size() - 2; i >= 0; i--)
		mNetworkLayers[i]->UpdateWeight(mLayerResults[i].get(), learningRate);
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