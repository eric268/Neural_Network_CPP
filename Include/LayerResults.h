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
		bool doOnce = false;
		for (int i = 0; i < mWeightedResults.size(); i++)
		{
			for (int j = 0; j < mWeightedResults[0].size(); j++)
			{
				if (!doOnce)
				{
					mBiasResults[j] += obj.mBiasResults[j];
				}
				mWeightedResults[i][j] += obj.mWeightedResults[i][j];
			}
			doOnce = true;
		}
		
		return {mWeightedResults, mBiasResults};
	}
	LayerResults operator *(double val)
	{
		bool doOnce = false;
		for (int i = 0; i < mWeightedResults.size(); i++)
		{
			for (int j = 0; j < mWeightedResults[0].size(); j++)
			{
				if (!doOnce)
				{
					mBiasResults[j] *= val;
				}
				mWeightedResults[i][j] *= val;
			}
			doOnce = true;
		}
		return {mWeightedResults, mBiasResults};
	}
};