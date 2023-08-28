#pragma once

class Neurons;
class LayerResults;

class NetworkLayer
{
public:
	NetworkLayer();
	explicit NetworkLayer(std::size_t numOfNeurons);
	NetworkLayer(std::size_t numOfNeurons, NetworkLayer* prevLayer, NetworkLayer* nextLayer);

	void UpdateBias(LayerResults* result, const double learningRate);
	void UpdateWeight(LayerResults* result, const double learningRate);

#pragma region Inline Functions
	inline const std::shared_ptr<NetworkLayer>& GetPreviousLayer() const		{ return previousLayer; }
	inline const std::shared_ptr<NetworkLayer>& GetNextLayer() const			{ return nextLayer; }
	inline const size_t GetLayerSize() const									{ return numberOfNeurons; }
	inline const std::vector<std::unique_ptr<Neurons>>& GetNeurons() const		{ return neurons; }
	inline const std::vector<std::vector<double>>& GetWeights()	const			{ return weights; }
	inline const std::vector<double>& GetBias()	const							{ return bias; }

	inline void SetPreviousLayer(const std::shared_ptr<NetworkLayer>& prev)		{ previousLayer = prev; }
	inline void SetNextLayer (const std::shared_ptr<NetworkLayer>& next)		{ nextLayer = next; }
	inline void SetWeights	(const std::vector<std::vector<double>>& w)			{ weights = w; }
	inline void SetBias (const std::vector<double>& b)							{ bias = b; }
#pragma endregion

private:
	size_t numberOfNeurons;
	std::shared_ptr<NetworkLayer> previousLayer;
	std::shared_ptr<NetworkLayer> nextLayer;
	std::vector<std::unique_ptr<Neurons>> neurons;
	std::vector<std::vector<double>> weights;
	std::vector<double> bias;

};