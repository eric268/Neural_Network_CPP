#include "../Include/pch.h"
#include "../Include/Connections.h"

Connections::Connections() : mNeuron{nullptr}, mActivation{0.0}
{
	//Want to start by randomizing all weights
	mWeight = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}