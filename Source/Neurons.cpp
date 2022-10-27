#include "../Include/pch.h"
#include "../Include/Neurons.h"
#include "../Include/MathHelper.h"
#include "../Include/Connections.h"
#include "../Include/NetworkLayer.h"

Neurons::Neurons() :mActivation{ 0.0 }, mBias{ 0.0 }, mZ{ 0.0 } {}

Neurons::Neurons(double val) :mActivation{ val }, mBias{ 0.0 }, mZ{ 0.0 } {}


void Neurons::UpdateNeuron()
{

}

void Neurons::PopulateConnections(NetworkLayer* nextLayer)
{
	int sizeOfNextLayer = nextLayer->mNumberOfNeurons;
	mConnections = std::vector<Connections*>(sizeOfNextLayer);
	for (int i = 0; i < sizeOfNextLayer; i++)
	{
		mConnections[i] = new Connections();
		mConnections[i]->mNeuron = nextLayer->mNeurons[i];
	}
}