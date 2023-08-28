#pragma once

class NeuralNetwork;
class DataManager;
class DisplayManager;
class HyperParameters;


class ApplicationManager
{
public:
	ApplicationManager() = default;
	ApplicationManager(std::unique_ptr<NeuralNetwork> network, std::unique_ptr<HyperParameters> parameters);
	~ApplicationManager();
	void Run();
	std::string GetMenuInput();
	void StartModelTraining();
	void StartModelTest();
	void TrainNetwork(const std::vector<std::pair<std::vector<double>, int>>& imageData);
	void TestNetwork(const std::vector<std::pair<std::vector<double>, int>>& imageData);
	void DisplayPredictions();
	void LoadNetwork();
	void SaveNetwork();

private:
	bool CheckIfValidFilename(const std::string& filename);

	std::unique_ptr <NeuralNetwork> neuralNetwork;
	std::unique_ptr <DataManager> dataManager;
	std::unique_ptr <DisplayManager>  displayManager;
	std::unique_ptr <HyperParameters> hyperParameters;

	int currentEpoch;
	int currentBatch;
	long double averageAccuracy;
	long double averageLoss;
};

