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

public:
	LayerType mLayerType;
	double mActivation;

	double mDeltaBias;
	double mDeltaError;
	double mDeltaOutput;
};