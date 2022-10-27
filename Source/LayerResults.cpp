#include "../Include/pch.h"
#include "../Include/LayerResults.h"

LayerResults::LayerResults(int previousLayerSize, int currentLayerSize)
{
	mWeightedResults = std::vector<std::vector<double>>(previousLayerSize, std::vector<double>(currentLayerSize));
	mBiasResults = std::vector<double>(currentLayerSize);
}