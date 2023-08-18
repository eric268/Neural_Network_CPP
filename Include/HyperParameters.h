#pragma once

struct HyperParameters
{
	HyperParameters(double learningRate, int numEpochs, int batchSize);

	int batchSize;
	int numEpochs;
	double learningRate;
};

