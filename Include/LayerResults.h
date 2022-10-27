#include "pch.h"

class LayerResults
{
public:
	LayerResults() = default;
	LayerResults(int previousLayerSize, int currentLayerSize);
	LayerResults(std::vector<std::vector<double>> weight, std::vector<double>bias) : mWeightedResults{ weight }, mBiasResults{ bias } {}
	std::vector<std::vector<double>> mWeightedResults;
	std::vector<double> mBiasResults;

	LayerResults operator+=(LayerResults obj)
	{
		for (int i = 0; i < mWeightedResults[0].size(); i++)
		{
			obj.mBiasResults[i] += mBiasResults[i];
			for (int j = 0; j < mWeightedResults.size(); j++)
			{
				obj.mWeightedResults[j][i] += mWeightedResults[j][i];
			}
		}
		return *this;
	}
	LayerResults operator /=(double val)
	{
		for (int i = 0; i < mWeightedResults[0].size(); i++)
		{
			mBiasResults[i] /= val;
			for (int j = 0; j < mWeightedResults.size(); j++)
			{
				mWeightedResults[j][i] /= val;
			}
		}
		return *this;
	}
};