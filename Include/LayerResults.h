#include "pch.h"

class LayerResults
{
public:
	LayerResults() = default;
	LayerResults(int previousLayerSize, int currentLayerSize);
	LayerResults(std::vector<std::vector<double>> weight, std::vector<double>bias) : mWeightedResults{ weight }, mBiasResults{ bias } {}
	std::vector<std::vector<double>> mWeightedResults;
	std::vector<double> mBiasResults;

	LayerResults operator+ (LayerResults obj)
	{
		for (int i = 0; i < mWeightedResults.size(); i++)
		{
			mBiasResults[i] += obj.mBiasResults[i];
			for (int j = 0; j < mWeightedResults[0].size(); j++)
			{
				mWeightedResults[i][j] += obj.mWeightedResults[i][j];
			}
		}
		
		return {mWeightedResults, mBiasResults};
	}

	LayerResults operator- (LayerResults obj)
	{
		for (int i = 0; i < mWeightedResults.size(); i++)
		{
			mBiasResults[i] -= obj.mBiasResults[i];
			for (int j = 0; j < mWeightedResults[0].size(); j++)
			{
				mWeightedResults[i][j] -= obj.mWeightedResults[i][j];
			}
		}

		return { mWeightedResults, mBiasResults };
	}

	LayerResults operator *(double val)
	{
		for (int i = 0; i < mWeightedResults.size(); i++)
		{
			mBiasResults[i] *= val;
			for (int j = 0; j < mWeightedResults[0].size(); j++)
			{
				mWeightedResults[i][j] *= val;
			}
		}
		return {mWeightedResults, mBiasResults};
	}
};