#pragma once

class HyperParameters
{
public:
	HyperParameters() = default;
	HyperParameters(int batchSize, int epochs, double learningRate);
	~HyperParameters() = default;

#pragma region Inline Getters
	const int GetBatchSize() const			{ return batchSize; }
	const int GetNumEpochs() const			{ return numEpochs; }
	const double GetLearningRate() const	{ return learningRate; }
#pragma endregion

private:
	int batchSize;
	int numEpochs;
	double learningRate;
};

