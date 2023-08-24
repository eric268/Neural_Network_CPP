#include "pch.h"

class LayerResults
{
public:
	LayerResults() = default;
	LayerResults(int previousLayerSize, int currentLayerSize);
	LayerResults(std::vector<std::vector<double>> weight, std::vector<double>bias) : weightedResults{ weight }, biasResults{ bias } {}
	std::vector<std::vector<double>> weightedResults;
	std::vector<double> biasResults;

	LayerResults operator+ (LayerResults obj)
	{
		for (int i = 0; i < weightedResults.size(); i++)
		{
			biasResults[i] += obj.biasResults[i];
			for (int j = 0; j < weightedResults[0].size(); j++)
			{
				weightedResults[i][j] += obj.weightedResults[i][j];
			}
		}
		
		return {weightedResults, biasResults};
	}

	LayerResults operator- (LayerResults obj)
	{
		for (int i = 0; i < weightedResults.size(); i++)
		{
			biasResults[i] -= obj.biasResults[i];
			for (int j = 0; j < weightedResults[0].size(); j++)
			{
				weightedResults[i][j] -= obj.weightedResults[i][j];
			}
		}

		return { weightedResults, biasResults };
	}

	LayerResults operator *(double val)
	{
		for (int i = 0; i < weightedResults.size(); i++)
		{
			biasResults[i] *= val;
			for (int j = 0; j < weightedResults[0].size(); j++)
			{
				weightedResults[i][j] *= val;
			}
		}
		return {weightedResults, biasResults};
	}
};