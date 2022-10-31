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
	Neurons(double val, LayerType type);


	double CalculateCost(bool isCorrect);
	double DCalculateCost(bool isCorrect);

	void PopulateConnections(NetworkLayer* nextLayer);

public:
	LayerType mLayerType;
	double mActivation;
	double mBias;

	double mDeltaBias;
	double mDeltaError;
	double mDeltaOutput;
};

/*what do I need for backwards propigations
 - y (done)
 - weight and activation are different!

*/