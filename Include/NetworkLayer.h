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
	std::shared_ptr<NetworkLayer> mPreviousLayer;
	std::shared_ptr<NetworkLayer> mNextLayer;
	int mNumberOfNeurons;
	std::vector<std::unique_ptr<Neurons>> mNeurons;
	std::vector<std::vector<double>> mWeights;
	void UpdateBias(LayerResults* result, double learningRate);
	void UpdateWeight(LayerResults* result, double learningRate);
};