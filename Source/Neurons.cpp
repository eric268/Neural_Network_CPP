#include "../Include/pch.h"
#include "../Include/Neurons.h"
#include "../Include/NetworkLayer.h"

Neurons::Neurons() : mLayerType{ LayerType::InputLayer }, mActivation{ 0.0 }, mBias{ 0.0 }, mDeltaBias{ 0.0 }, mDeltaError{ 0.0 }, mDeltaOutput{ 0.0 }
{
	std::random_device rd;
	std::uniform_int_distribution<int> dist(-10000, 10000);
	mBias = ((double)dist(rd)) * 0.00001;
}

Neurons::Neurons(LayerType type) : mLayerType{ type }, mActivation{ 0.0 }, mBias{ 0.0 }, mDeltaBias{ 0.0 }, mDeltaError{ 0.0 }, mDeltaOutput{ 0.0 }
{
}

Neurons::Neurons(double val, LayerType type) :mActivation{ val }, mLayerType{ type }, mBias{ 0.0 }, mDeltaBias{ 0.0 }, mDeltaError{ 0.0 }, mDeltaOutput{ 0.0 } {}