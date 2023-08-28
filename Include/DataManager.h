#pragma once

class NetworkLayer;

class DataManager
{
public:
	DataManager();
	~DataManager() = default;

	const std::vector<std::pair<std::vector<double>, int>> LoadImageData(std::string path, std::string labelsPath, int numberOfImages);
	int ReverseInt(int i);
	void ShuffleTrainingData();

#pragma region Inline Functions
	inline const std::vector<std::pair<std::vector<double>, int>> GetTrainingData() const	{ return trainingData; }
	inline const std::vector<std::pair<std::vector<double>, int>> GetTestingData() const	{ return testingData;  }
#pragma endregion

private:
	std::vector<std::pair<std::vector<double>, int>> trainingData;
	std::vector<std::pair<std::vector<double>, int>> testingData;
};

