#include "../Include/pch.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/LayerResults.h"

NetworkLayer::NetworkLayer()
{
	mNumberOfNeurons = 0;
	mPreviousLayer = nullptr;
	mNextLayer = nullptr;
}

NetworkLayer::NetworkLayer(int numofNeurons) :  mNumberOfNeurons{numofNeurons}
{
	mPreviousLayer = nullptr;
	mNextLayer = nullptr;
	mNeurons = std::vector<std::unique_ptr<Neurons>>(mNumberOfNeurons);
	for (int i = 0; i < mNumberOfNeurons; i++)
	{
		mNeurons[i] = std::make_unique<Neurons>();
	}
}

NetworkLayer::NetworkLayer(int numOfNeurons, NetworkLayer* prevLayer, NetworkLayer* nextLayer) : mNumberOfNeurons{numOfNeurons}, mPreviousLayer{prevLayer}, mNextLayer{nextLayer}
{
	mNeurons = std::vector<std::unique_ptr<Neurons>>(mNumberOfNeurons);
	for (int i = 0; i < mNumberOfNeurons; i++)
	{
		mNeurons[i] = std::make_unique<Neurons>();
	}
}

void NetworkLayer::UpdateBias(LayerResults* result, double learningRate)
{
	for (int i = 0; i < mNumberOfNeurons; i++)
	{
		mNeurons[i]->mBias -= result->mBiasResults[i] * learningRate;
	}
}

void NetworkLayer::UpdateWeight(LayerResults* result, double learningRate)
{
	for (int i = 0; i < mWeights.size(); i++)
	{
		for (int j = 0; j < mWeights[0].size(); j++)
		{
			mWeights[i][j] -= result->mWeightedResults[i][j] * learningRate;
		}
	}
}

