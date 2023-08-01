#include "../Include/pch.h"
#include "../Include/LayerResults.h"

LayerResults::LayerResults(int currentLayerSize, int previousLayerSize)
{
	mWeightedResults = std::vector<std::vector<long float>>(currentLayerSize, std::vector<long float>(previousLayerSize));
	mBiasResults = std::vector<long float>(currentLayerSize);
}