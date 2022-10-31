#include "../Include/pch.h"
#include "../Include/Neurons.h"
#include "../Include/MathHelper.h"
#include "../Include/Connections.h"
#include "../Include/NetworkLayer.h"

Neurons::Neurons() : mLayerType{ LayerType::InputLayer }, mActivation{ 0.0 }, mBias{ 0.0 }, mDeltaBias{ 0.0 }, mDeltaError{ 0.0 }, mDeltaOutput{ 0.0 } {}

Neurons::Neurons(LayerType type) : mLayerType{ type }, mActivation{ 0.0 }, mBias{ 0.0 }, mDeltaBias{ 0.0 }, mDeltaError{ 0.0 }, mDeltaOutput{ 0.0 }
{
}

Neurons::Neurons(double val, LayerType type) :mActivation{ val }, mLayerType{ type }, mBias{ 0.0 }, mDeltaBias{ 0.0 }, mDeltaError{ 0.0 }, mDeltaOutput{ 0.0 } {}


//void Neurons::PopulateConnections(NetworkLayer* nextLayer)
//{
//	int sizeOfNextLayer = nextLayer->mNumberOfNeurons;
//	mConnections = std::vector<Connections*>(sizeOfNextLayer);
//	for (int i = 0; i < sizeOfNextLayer; i++)
//	{
//		mConnections[i] = new Connections();
//		mConnections[i]->mNeuron = nextLayer->mNeurons[i];
//	}
//}