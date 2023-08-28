#pragma once

class LayerResults
{
public:
	LayerResults() = default;
	LayerResults(int previousLayerSize, int currentLayerSize);
	LayerResults(std::vector<std::vector<double>> weight, std::vector<double>bias);
	~LayerResults() = default;
	void ClearResults();

	LayerResults operator+ (LayerResults obj);
	LayerResults operator- (LayerResults obj);
	LayerResults operator* (double val);

#pragma region Inline Getters & Setters
	inline const std::vector<std::vector<double>>& GetWeightResults() const			{ return weightedResults; }
	inline const std::vector<double>& GetBiasResults() const						{ return biasResults; }

	inline void SetWeightResults(const std::vector<std::vector<double>>& results)	{ weightedResults = results; }
	inline void SetBiasResults(const std::vector<double>& results)					{ biasResults = results; }
#pragma endregion

private:
	std::vector<std::vector<double>> weightedResults;
	std::vector<double> biasResults;
};