#pragma once
#include "pch.h"

class NetworkLayer;

class DataManager
{
public:
	DataManager();
	~DataManager();

	int ReverseInt(int i);
	const std::vector<std::pair<std::vector<double>, int>> LoadImageData(std::string path, std::string labelsPath, int numberOfImages);
	void ShuffleTrainingData();

#pragma region Inline Getters & Setters
	const std::vector<std::pair<std::vector<double>, int>> GetTrainingData() const { return trainingData; }
	const std::vector<std::pair<std::vector<double>, int>> GetTestingData() const  { return testingData;  }
#pragma endregion

private:
	std::vector<std::pair<std::vector<double>, int>> trainingData;
	std::vector<std::pair<std::vector<double>, int>> testingData;
};

