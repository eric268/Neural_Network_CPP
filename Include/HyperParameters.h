#pragma once

class HyperParameters
{
public:
	HyperParameters() = default;
	~HyperParameters() = default;
	HyperParameters(int batchSize, int epochs, double learningRate);

#pragma region Inline Functions
	inline const int GetBatchSize() const			{ return batchSize; }
	inline const int GetNumEpochs() const			{ return numEpochs; }
	inline const double GetLearningRate() const		{ return learningRate; }
#pragma endregion

private:
	int batchSize;
	int numEpochs;
	double learningRate;
};

