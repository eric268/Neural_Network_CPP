#include "../Include/pch.h"
#include "../Include/Connections.h"

Connections::Connections() : mNeuron{nullptr}, mActivation{0.0}
{
	//Want to start by randomizing all weights
	double low = -1.0;
	double high = 1.0;

	mWeight = low + static_cast<double>(rand()) / static_cast<double>(RAND_MAX / (high - low));
}