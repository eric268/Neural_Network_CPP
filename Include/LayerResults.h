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

	const std::vector<std::vector<double>>& GetWeightResults() const { return weightedResults; }
	const std::vector<double>& GetBiasResults() const { return biasResults; }

	void SetWeightResults(const std::vector<std::vector<double>>& results) { weightedResults = results; }
	void SetBiasResults(const std::vector<double>& results) { biasResults = results; }
private:
	std::vector<std::vector<double>> weightedResults;
	std::vector<double> biasResults;
};