#pragma once

class NeuralNetwork;
class DataManager;

class DisplayManager
{
public:
	DisplayManager();
	~DisplayManager() = default;
	void DisplayMainMenu();
	bool DrawPredictionsMenu();
	void DrawNetworkPredictions(NeuralNetwork* network, DataManager* dataManager);
	std::string GetNumberDisplay(const std::pair<std::vector<double>, int>& imageData, const int networkPrediction);
	void DisplayResults(std::string results);
	std::string ParseResults(const int currentEpoch, const int maxEpoch,const int currentBatch, const int totalBatches, const long double loss, const long double accuracy);
	static void ClearConsole();

#pragma region Inline Functions
	inline void SaveResults(const int currentEpoch, const int maxEpoch, const int currentBatch, const int totalBatches, const long double loss, const long double accuracy)
	{
		epochResults.push_back(ParseResults(currentEpoch, maxEpoch, currentBatch, totalBatches, loss, accuracy));
	}
	inline void ClearResults() 
	{ 
		epochResults.clear();
	}
#pragma endregion

private:
	int numImagesToDisplay;
	std::vector<std::string> epochResults;
};

