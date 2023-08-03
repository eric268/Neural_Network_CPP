#pragma once
#include "pch.h"

class DataManager
{
public:
	DataManager();
	~DataManager();

	int ReverseInt(int i);
	std::vector<std::pair<std::vector<double>, int>> LoadImageData(std::string path, std::string labelsPath, int numberOfImages);
	void ShuffleTrainingData();

	std::vector<std::pair<std::vector<double>, int>> trainingData;
	std::vector<std::pair<std::vector<double>, int>> testingData;

private:

};

