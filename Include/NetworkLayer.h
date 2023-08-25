#pragma once

class Neurons;
class LayerResults;

class NetworkLayer
{
public:
	NetworkLayer();
	explicit NetworkLayer(std::size_t numofNeurons);
	NetworkLayer(std::size_t numOfNeurons, NetworkLayer* prevLayer, NetworkLayer* nextLayer);
	void UpdateBias(LayerResults* result, double learningRate);
	void UpdateWeight(LayerResults* result, double learningRate);

#pragma region Inline Getters & Setters
	const std::shared_ptr<NetworkLayer>&		 GetPreviousLayer() const { return previousLayer; }
	const std::shared_ptr<NetworkLayer>&		 GetNextLayer()		const { return nextLayer; }
	const size_t								 GetLayerSize()		const { return numberOfNeurons; }
	const std::vector<std::unique_ptr<Neurons>>& GetNeurons()		const { return neurons; }
	const std::vector<std::vector<double>>&		 GetWeights()		const { return weights; }
	const std::vector<double>&					 GetBias()			const { return bias; }

	void SetPreviousLayer(const std::shared_ptr<NetworkLayer>& prev)		 { previousLayer = prev; }
	void SetNextLayer		(const std::shared_ptr<NetworkLayer>& next)		 { nextLayer = next; }
	void SetWeights			(const std::vector<std::vector<double>>& w)		 { weights = w; }
	void SetBias			(const std::vector<double>& b)					 { bias = b; }
#pragma endregion

private:
	size_t numberOfNeurons;
	std::shared_ptr<NetworkLayer> previousLayer;
	std::shared_ptr<NetworkLayer> nextLayer;
	std::vector<std::unique_ptr<Neurons>> neurons;
	std::vector<std::vector<double>> weights;
	std::vector<double> bias;

};