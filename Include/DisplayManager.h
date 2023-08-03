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
private:

};

