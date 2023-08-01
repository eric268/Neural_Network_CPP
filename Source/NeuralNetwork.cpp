#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include <thread>

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes) : mTotalError{ 0.0 }
{
	assert(layerSizes.size() > 0);

	mNetworkLayers = std::vector<NetworkLayer*>(layerSizes.size());
	mNetworkLayers[0] = new NetworkLayer(layerSizes[0]);
	mLayerResults = std::vector<LayerResults*>(layerSizes.size() - 1);

	for (int i = 1; i < layerSizes.size(); i++)
	{
		mNetworkLayers[i] = new NetworkLayer(layerSizes[i]);
		mNetworkLayers[i - 1]->mNextLayer = mNetworkLayers[i];
		mNetworkLayers[i]->mPreviousLayer = mNetworkLayers[i - 1];
		mLayerResults[i - 1] = new LayerResults(layerSizes[i], layerSizes[i - 1]);
		PopulateNeuronsInLayers(mNetworkLayers[i - 1]);
	}
}

void NeuralNetwork::PopulateNeuronsInLayers(NetworkLayer* currentLayer)
{
	if (!currentLayer || !currentLayer->mNextLayer)
		return;

	currentLayer->mWeights = std::vector<std::vector<float>>(currentLayer->mNextLayer->mNumberOfNeurons, std::vector<float>(currentLayer->mNumberOfNeurons));
	std::random_device rd;
	std::uniform_int_distribution<int> dist(-10000, 10000);

	for (int i = 0; i < currentLayer->mNextLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->mNumberOfNeurons; j++)
		{
			float val = ((float)dist(rd)) * 0.00001;
			currentLayer->mWeights[i][j] = val;
		}
	}
}

void NeuralNetwork::TrainNetwork()
{

}

void NeuralNetwork::TestNetwork()
{

}

void NeuralNetwork::ClearResults()
{
	mTotalError = 0.0;

	for (int i = 0; i < mLayerResults.size(); i++)
	{
		std::fill(mLayerResults[i]->mBiasResults.begin(), mLayerResults[i]->mBiasResults.end(), 0.0);

		for (auto& p :mLayerResults[i]->mWeightedResults)
		{
			std::fill(p.begin(), p.end(), 0.0);
		}
	}
}

int NeuralNetwork::RunNetwork(std::vector<float> pixelValues)
{
	for (int i = 0; i < pixelValues.size(); i++)
		mNetworkLayers[0]->mNeurons[i]->mActivation = pixelValues[i] / 255.0;

	for (int i = 0; i < mNetworkLayers.size() - 1; i++)
		SetNextLayersActivation(mNetworkLayers[i]);

	return GetFinalOutput(mNetworkLayers[mNetworkLayers.size() - 1]);
}

int NeuralNetwork::GetFinalOutput(NetworkLayer* outputLayer)
{
	float highestActivation = -INFINITY;
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
	for (int i = 0; i < currentLayer->mNextLayer->mNumberOfNeurons; i++)
	{
		currentLayer->mNextLayer->mNeurons[i]->mActivation = 0.0;
		for (int j = 0; j < currentLayer->mNumberOfNeurons; j++)
		{
			currentLayer->mNextLayer->mNeurons[i]->mActivation += currentLayer->mWeights[i][j] * currentLayer->mNeurons[j]->mActivation;
		}
		currentLayer->mNextLayer->mNeurons[i]->mActivation = MathHelper::Sigmoid(currentLayer->mNextLayer->mNeurons[i]->mActivation +
			currentLayer->mNextLayer->mNeurons[i]->mBias);
	}
}

void NeuralNetwork::CalculateLayerDeltaCost(int correctAns)
{
	//Output layer backprop is different function from other layers so is done seperatly 
	CalculateOutputLayerBackwardsProp(mNetworkLayers[mNetworkLayers.size() - 1], mLayerResults[mLayerResults.size() - 1], correctAns);

	for (int i = mNetworkLayers.size() - 2; i > 0; i--)
		CalculateLayerBackwardsPropigation(mNetworkLayers[i], mLayerResults[i - 1], correctAns);
}

void NeuralNetwork::CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* layerResults, int correctAns)
{
	float y = 0.0;
	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		y = (correctAns == i) ? 1.0 : 0.01;
		mTotalError += (currentLayer->mNeurons[i]->mActivation - y) * (currentLayer->mNeurons[i]->mActivation - y);
		currentLayer->mNeurons[i]->mDeltaError = 2.0 * (currentLayer->mNeurons[i]->mActivation - y);
		currentLayer->mNeurons[i]->mDeltaOutput = currentLayer->mNeurons[i]->mActivation * (1.0 - currentLayer->mNeurons[i]->mActivation);
		layerResults->mBiasResults[i] = currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput;

		for (int j = 0; j < currentLayer->mPreviousLayer->mNumberOfNeurons; j++)
		{
			layerResults->mWeightedResults[i][j] += currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput * currentLayer->mPreviousLayer->mNeurons[j]->mActivation;
		}
	}
}

void NeuralNetwork::CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, LayerResults* layerResults, int correctAns)
{
	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		currentLayer->mNeurons[i]->mDeltaError = 0.0;
		currentLayer->mNeurons[i]->mDeltaOutput = currentLayer->mNeurons[i]->mActivation * (1.0 - currentLayer->mNeurons[i]->mActivation);
		for (int j = 0; j < currentLayer->mNextLayer->mNumberOfNeurons; j++)
		{
			currentLayer->mNeurons[i]->mDeltaError += currentLayer->mNextLayer->mNeurons[j]->mDeltaError
				* currentLayer->mNextLayer->mNeurons[j]->mDeltaOutput * currentLayer->mWeights[j][i];
		}
		layerResults->mBiasResults[i] = currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput;
	}

	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->mPreviousLayer->mNumberOfNeurons; j++)
		{
			layerResults->mWeightedResults[i][j] += currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput * currentLayer->mPreviousLayer->mNeurons[j]->mActivation;
		}
	}
}

void NeuralNetwork::UpdateResults(int testSize)
{
	system("CLS");
	for (int i = 1; i < mNetworkLayers.size(); i++)
		mNetworkLayers[i]->UpdateBias(mLayerResults[i - 1], learningRate);

	for (int i = mNetworkLayers.size() - 2; i >= 0; i--)
		mNetworkLayers[i]->UpdateWeight(mLayerResults[i], learningRate);
}