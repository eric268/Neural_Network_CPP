#include "../Include/pch.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"

NetworkLayer::NetworkLayer()
{
	mNumberOfNeurons = 0;
	mPreviousLayer = nullptr;
	mNextLayer = nullptr;
}

NetworkLayer::NetworkLayer(LayerType type, int numofNeurons) :  mNumberOfNeurons{numofNeurons}
{
	mPreviousLayer = nullptr;
	mNextLayer = nullptr;
	mNeurons = std::vector<Neurons*>(mNumberOfNeurons);
	for (int i = 0; i < mNumberOfNeurons; i++)
	{
		mNeurons[i] = new Neurons(type);
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

