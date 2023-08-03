#include "../Include/pch.h"
#include "../Include/HyperParameters.h"

HyperParameters::HyperParameters(double learningRate, int numEpochs, int batchSize) :
	learningRate(learningRate),
	numEpochs(numEpochs), 
	batchSize(batchSize)
{
}