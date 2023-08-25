#pragma once

class LayerResults
{
public:
	LayerResults() = default;
	LayerResults(int previousLayerSize, int currentLayerSize);
	LayerResults(std::vector<std::vector<double>> weight, std::vector<double>bias) : weightedResults{ weight }, biasResults{ bias } {}

	LayerResults operator+ (LayerResults obj)
	{
		for (int i = 0; i < weightedResults.size(); i++)
		{
			biasResults[i] += obj.biasResults[i];
			for (int j = 0; j < weightedResults[0].size(); j++)
			{
				weightedResults[i][j] += obj.weightedResults[i][j];
			}
		}
		
		return {weightedResults, biasResults};
	}

	LayerResults operator- (LayerResults obj)
	{
		for (int i = 0; i < weightedResults.size(); i++)
		{
			biasResults[i] -= obj.biasResults[i];
			for (int j = 0; j < weightedResults[0].size(); j++)
			{
				weightedResults[i][j] -= obj.weightedResults[i][j];
			}
		}

		return { weightedResults, biasResults };
	}

	LayerResults operator *(double val)
	{
		for (int i = 0; i < weightedResults.size(); i++)
		{
			biasResults[i] *= val;
			for (int j = 0; j < weightedResults[0].size(); j++)
			{
				weightedResults[i][j] *= val;
			}
		}
		return {weightedResults, biasResults};
	}
	void ClearResults();

	inline const std::vector<std::vector<double>>& GetWeightResults() const { return weightedResults; }
	inline const std::vector<double>& GetBiasResults() const { return biasResults; }

	inline void SetWeightResults(const std::vector<std::vector<double>>& results) { weightedResults = results; }
	inline void SetBiasResults(const std::vector<double>& results) { biasResults = results; }
private:
	std::vector<std::vector<double>> weightedResults;
	std::vector<double> biasResults;
};