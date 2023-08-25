#include "../Include/pch.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/LayerResults.h"

NetworkLayer::NetworkLayer() : numberOfNeurons{0}, previousLayer{nullptr}, nextLayer{nullptr} {}

NetworkLayer::NetworkLayer(std::size_t numofNeurons) : numberOfNeurons{ numofNeurons }, nextLayer{nullptr}, previousLayer{nullptr}
{
	neurons = std::vector<std::unique_ptr<Neurons>>(numberOfNeurons);
	for (int i = 0; i < numberOfNeurons; i++)
	{
		neurons[i] = std::make_unique<Neurons>();
	}
}

NetworkLayer::NetworkLayer(std::size_t numOfNeurons, NetworkLayer* prevLayer, NetworkLayer* nextLayer) : numberOfNeurons{ numOfNeurons }, previousLayer{ prevLayer }, nextLayer{ nextLayer }
{
	neurons = std::vector<std::unique_ptr<Neurons>>(numberOfNeurons);
	for (int i = 0; i < numberOfNeurons; i++)
	{
		neurons[i] = std::make_unique<Neurons>();
	}
}

void NetworkLayer::UpdateBias(LayerResults* result, double learningRate)
{
	if (!result || result->GetBiasResults().size() != bias.size())
		throw std::invalid_argument("Invalid layer result, or invalid size\n");

	for (int i = 0; i < result->GetBiasResults().size(); i++)
	{
		bias[i] -= result->GetBiasResults()[i] * learningRate;
	}
}

void NetworkLayer::UpdateWeight(LayerResults* result, double learningRate)
{
	if (!result													|| 
		!weights.size()											||
		result->GetWeightResults().size() != weights.size()		|| 
		result->GetWeightResults()[0].size() != weights[0].size()
		)
		throw std::invalid_argument("Invalid layer result, or invalid size\n");

	for (int i = 0; i < weights.size(); i++)
	{
		for (int j = 0; j < weights[0].size(); j++)
		{
			weights[i][j] -= result->GetWeightResults()[i][j] * learningRate;
		}
	}
}

