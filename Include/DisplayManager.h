#pragma once

class NeuralNetwork;
class DataManager;

class DisplayManager
{
public:
	DisplayManager();
	std::string UserInputOnTrainingCompleted();
	void DrawNetworkPredictions(NeuralNetwork& network, DataManager& dataManager);
	void DrawNumber(const std::pair<std::vector<double>, int>& imageData, const int networkPrediction);

	int numImagesToDisplay;

	std::vector<std::string> epochResults;

	void DisplayResults(std::string results);
	std::string ParseResults(const int currentEpoch, const int maxEpoch,const int currentBatch, const int totalBatches, const long double loss, const long double accuracy);
	inline void SaveResults(const int currentEpoch, const int maxEpoch, const int currentBatch, const int totalBatches, const long double loss, const long double accuracy)
	{
		epochResults.push_back(ParseResults(currentEpoch, maxEpoch, currentBatch, totalBatches, loss, accuracy));
	}

	static void ClearConsole();
private:

};

