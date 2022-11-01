#include "pch.h"

enum LayerType
{
	InputLayer,
	HiddenLayer1,
	HiddenLayer2,
	OutputLayer
};

class Neurons;
class LayerResults;
class NetworkLayer
{
public:
	NetworkLayer();
	NetworkLayer(int numofNeurons);
	NetworkLayer(int numOfNeurons, NetworkLayer* prevLayer, NetworkLayer* nextLayer);
	NetworkLayer* mPreviousLayer;
	NetworkLayer* mNextLayer;
	int mNumberOfNeurons;
	std::vector<Neurons*> mNeurons;
	std::vector<std::vector<double>> mWeights;
	void UpdateBias(LayerResults* result);
	void UpdateWeight(LayerResults* result);
};