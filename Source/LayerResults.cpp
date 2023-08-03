#include "../Include/pch.h"
#include "../Include/LayerResults.h"

LayerResults::LayerResults(int currentLayerSize, int previousLayerSize)
{
	mWeightedResults = std::vector<std::vector<double>>(currentLayerSize, std::vector<double>(previousLayerSize));
	mBiasResults = std::vector<double>(currentLayerSize);
}