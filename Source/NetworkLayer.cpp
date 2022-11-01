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
	mNeurons = std::vector<Neurons*>(mNumberOfNeurons);
	for (int i = 0; i < mNumberOfNeurons; i++)
	{
		mNeurons[i] = new Neurons();
	}
}

NetworkLayer::NetworkLayer(int numOfNeurons, NetworkLayer* prevLayer, NetworkLayer* nextLayer) : mNumberOfNeurons{numOfNeurons}, mPreviousLayer{prevLayer}, mNextLayer{nextLayer}
{
	mNeurons = std::vector<Neurons*>(mNumberOfNeurons);
	for (int i = 0; i < mNumberOfNeurons; i++)
	{
		mNeurons[i] = new Neurons();
	}
}

void NetworkLayer::UpdateBias(LayerResults* result)
{
	for (int i = 0; i < mWeights.size(); i++)
	{
		mNeurons[i]->mBias -= result->mBiasResults[i];
	}
}

void NetworkLayer::UpdateWeight(LayerResults* result)
{
	for (int i = 0; i < mWeights.size(); i++)
	{
		for (int j = 0; j < mWeights[0].size(); j++)
		{
			mWeights[i][j] -= result->mWeightedResults[i][j];
		}
	}
}

