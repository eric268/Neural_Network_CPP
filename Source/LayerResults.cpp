#include "../Include/pch.h"
#include "../Include/LayerResults.h"

LayerResults::LayerResults(int currentLayerSize, int previousLayerSize)
{
	weightedResults = std::vector<std::vector<double>>(currentLayerSize, std::vector<double>(previousLayerSize));
	biasResults = std::vector<double>(currentLayerSize);
}

void LayerResults::ClearResults()
{
	std::fill(biasResults.begin(), biasResults.end(), 0.0);
	for (auto& w : weightedResults)
	{
		std::fill(w.begin(), w.end(), 0.0);
	}
}