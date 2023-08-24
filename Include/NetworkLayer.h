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
	void UpdateBias(LayerResults* result, double learningRate);
	void UpdateWeight(LayerResults* result, double learningRate);

	std::shared_ptr<NetworkLayer> previousLayer;
	std::shared_ptr<NetworkLayer> nextLayer;
	int numberOfNeurons;
	std::vector<std::unique_ptr<Neurons>> neurons;
	std::vector<std::vector<double>> weights;
	std::vector<double> bias;

};