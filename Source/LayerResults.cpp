#include "../Include/pch.h"
#include "../Include/LayerResults.h"

LayerResults::LayerResults(int currentLayerSize, int previousLayerSize) :
	weightedResults( std::vector<std::vector<double>>(currentLayerSize, std::vector<double>(previousLayerSize))),
	biasResults(std::vector<double>(currentLayerSize))
{}

LayerResults::LayerResults(std::vector<std::vector<double>> weight, std::vector<double>bias) :
	weightedResults{ weight }, 
	biasResults{ bias } 
{}

void LayerResults::ClearResults()
{
	std::fill(biasResults.begin(), biasResults.end(), 0.0);
	for (auto& w : weightedResults)
	{
		std::fill(w.begin(), w.end(), 0.0);
	}
}

LayerResults LayerResults::operator+ (LayerResults obj)
{
	for (int i = 0; i < weightedResults.size(); i++)
	{
		biasResults[i] += obj.biasResults[i];
		for (int j = 0; j < weightedResults[0].size(); j++)
		{
			weightedResults[i][j] += obj.weightedResults[i][j];
		}
	}
	return { weightedResults, biasResults };
}

LayerResults LayerResults::operator- (LayerResults obj)
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

LayerResults LayerResults::operator *(double val)
{
	for (int i = 0; i < weightedResults.size(); i++)
	{
		biasResults[i] *= val;
		for (int j = 0; j < weightedResults[0].size(); j++)
		{
			weightedResults[i][j] *= val;
		}
	}
	return { weightedResults, biasResults };
}