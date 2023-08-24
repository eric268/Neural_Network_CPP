#include "../Include/pch.h"
#include "../Include/LayerResults.h"

LayerResults::LayerResults(int currentLayerSize, int previousLayerSize)
{
	weightedResults = std::vector<std::vector<double>>(currentLayerSize, std::vector<double>(previousLayerSize));
	biasResults = std::vector<double>(currentLayerSize);
}