#pragma once
#include "pch.h"


class Connections;
class NetworkLayer;
enum LayerType;

class Neurons
{
public:
	Neurons();
	Neurons(LayerType type);
	Neurons(long float val, LayerType type);


	long float CalculateCost(bool isCorrect);
	long float DCalculateCost(bool isCorrect);

	void PopulateConnections(NetworkLayer* nextLayer);

public:
	LayerType mLayerType;
	long float mActivation;
	long float mBias;

	long float mDeltaBias;
	long float mDeltaError;
	long float mDeltaOutput;
};