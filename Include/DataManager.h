#pragma once
#include "pch.h"

class NetworkLayer;

class DataManager
{
public:
	DataManager();
	~DataManager();

	int ReverseInt(int i);
	std::vector<std::pair<std::vector<double>, int>> LoadImageData(std::string path, std::string labelsPath, int numberOfImages);
	void ShuffleTrainingData();

	void SaveWeightsAndBias(NetworkLayer* firstHiddenLayer, const std::string& filename);
	void LoadWeightsAndBias(NetworkLayer* firstHiddenLayer, const std::string& filename);
	std::vector<std::pair<std::vector<double>, int>> trainingData;
	std::vector<std::pair<std::vector<double>, int>> testingData;

private:

};

