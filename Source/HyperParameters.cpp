#include "../Include/pch.h"
#include "../Include/HyperParameters.h"

HyperParameters::HyperParameters(int batchSize, int epochs, double learningRate) :
	batchSize(batchSize),
	numEpochs(epochs),
	learningRate(learningRate)
{
}