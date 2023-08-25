#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/ActivationFuncTypes.h"
#include "../Include/WeightInitializer.h"
#include "../Include/ActivationFunctions.h"
#include "../Include/DisplayManager.h"
#include "../Include/FileManager.h"

NeuralNetwork::NeuralNetwork(std::vector<int>& layerSizes, ActivationFunctionTypes type) : 
	totalLoss (0.0),
	activationFunctionType (type)
{
	assert(layerSizes.size() > 0);

	networkLayers = std::vector<std::shared_ptr<NetworkLayer>>(layerSizes.size());
	networkLayers[0] = std::make_shared<NetworkLayer>(layerSizes[0]);
	mLayerResults = std::vector<std::shared_ptr<LayerResults>>(layerSizes.size() - 1);

	for (int i = 1; i < layerSizes.size(); i++)
	{
		networkLayers[i] = std::make_shared<NetworkLayer>(layerSizes[i]);
		networkLayers[i - 1]->SetNextLayer(networkLayers[i]);
		networkLayers[i]->SetPreviousLayer(networkLayers[i - 1]);
		mLayerResults[i - 1] = std::make_shared<LayerResults>(layerSizes[i], layerSizes[i - 1]);
		PopulateNeuronsInLayers(networkLayers[i].get());
	}
	BindActivationFunctions(type);
}

void NeuralNetwork::PopulateNeuronsInLayers(NetworkLayer* currentLayer)
{
	if (!currentLayer || !currentLayer->GetPreviousLayer())
	{
		std::cerr << "Invalid layer or previous layer is nullptr.\n";
		throw std::invalid_argument("Invalid layer provided");
	}

	const size_t outputSize = currentLayer->GetLayerSize();
	const size_t inputSize = currentLayer->GetPreviousLayer()->GetLayerSize();

	currentLayer->SetWeights(InitalizeNetworkWeights(inputSize, outputSize));
	currentLayer->SetBias(InitalizeBias(currentLayer->GetLayerSize()));
}


void NeuralNetwork::ClearResults()
{
	totalLoss = 0.0;
	try
	{
		for (int i = 0; i < mLayerResults.size(); i++)
			mLayerResults[i]->ClearResults();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw;
	}
}

int NeuralNetwork::RunNetwork(std::vector<double> pixelValues)
{

	// Input Layer
	SetNetworkInputs(pixelValues);

	// Hidden Layers
	for (int i = 1; i < networkLayers.size() - 1; i++)
		SetHiddenLayersActivation(networkLayers[i].get());

	//Output Layer
	SetOutputLayerActivation(networkLayers[networkLayers.size() - 1].get());

	return GetFinalOutput(networkLayers[networkLayers.size() - 1].get());
}

void NeuralNetwork::SetNetworkInputs(std::vector<double> pixelValues)
{
	const size_t inputSize = pixelValues.size();
	assert(networkLayers.size() && networkLayers[0]->GetNeurons().size() == inputSize);

	try
	{
		std::vector<double> activation (inputSize, 0.0);
		for (int i = 0; i < inputSize; i++)
			activation[i] = pixelValues[i] / 255.0;

		// If this point is reached no exceptions were throw therefore update network layer
		for (int i =0; i < inputSize; i++)
			networkLayers[0]->GetNeurons()[i]->SetActivation(activation[i]);
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw;
	}
}

void NeuralNetwork::SetHiddenLayersActivation(NetworkLayer* currentLayer)
{
	if (!currentLayer || !currentLayer->GetPreviousLayer()) {
		std::cerr << "Invalid layer or previous layer is nullptr.\n";
		throw std::invalid_argument("Invalid layer provided");
	}

	try
	{
		const size_t currentLayerSize = currentLayer->GetLayerSize();
		const size_t prevLayerSize = currentLayer->GetPreviousLayer()->GetLayerSize();

		std::vector<double> newActivations(currentLayerSize, 0.0);
		for (int i = 0; i < currentLayerSize; i++)
		{
			double sumActivation = 0.0;
			for (int j = 0; j < prevLayerSize; j++)
			{
				sumActivation += currentLayer->GetWeights()[i][j] * currentLayer->GetPreviousLayer()->GetNeurons()[j]->GetActivation();
			}

			// Add bias and apply activation function
			newActivations[i] = ActivationFunction(sumActivation + currentLayer->GetBias()[i]);
		}

		// If this point is reached no exceptions were throw therefore update network layer
		for (int i = 0; i < currentLayerSize; i++) 
		{
			currentLayer->GetNeurons()[i]->SetActivation(newActivations[i]);
		}
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what();
		throw;
	}
}

void NeuralNetwork::SetOutputLayerActivation(NetworkLayer* outputLayer)
{
	if (!outputLayer || !outputLayer->GetPreviousLayer()) 
	{
		std::cerr << "Invalid layer or previous layer is nullptr.\n";
		throw std::invalid_argument("Invalid layer provided");
	}

	try 
	{
		const size_t currentLayerSize = outputLayer->GetLayerSize();
		const size_t prevLayerSize = outputLayer->GetPreviousLayer()->GetLayerSize();

		// Local variable to store the new activations temporarily
		std::vector<double> newActivations(currentLayerSize, 0.0);

		// Calculate activations for the current layer
		for (int i = 0; i < currentLayerSize; ++i) {
			double sumActivation = 0.0;

			for (int j = 0; j < prevLayerSize; ++j) {
				sumActivation += outputLayer->GetWeights()[i][j] * outputLayer->GetPreviousLayer()->GetNeurons()[j]->GetActivation();
			}

			// Add the bias
			sumActivation += outputLayer->GetBias()[i];
			newActivations[i] = sumActivation;
		}

		// Calculate softmax probabilities
		std::vector<double> probabilities = ActivationFunctions::softmax(newActivations);

		// If we reached here, all calculations were successful. Update the activations.
		for (int i = 0; i < currentLayerSize; ++i) {
			outputLayer->GetNeurons()[i]->SetActivation(probabilities[i]);
		}
	}
	catch (const std::exception& ex) {
		std::cerr << "An error occurred: " << ex.what() << '\n';
		throw;
	}
}

int NeuralNetwork::GetFinalOutput(NetworkLayer* outputLayer)
{
	if (!outputLayer )
	{
		throw std::invalid_argument("Invalid layer provided");
	}

	double highestActivation = -INFINITY;
	int i, ans = -1;
	for (i = 0; i < outputLayer->GetNeurons().size(); i++)
	{
		if (highestActivation < outputLayer->GetNeurons()[i]->GetActivation())
		{
			highestActivation = outputLayer->GetNeurons()[i]->GetActivation();
			ans = i;
		}
	}
	return ans;
}


void NeuralNetwork::CalculateLayerDeltaCost(int correctAns)
{
	//Output layer back prop is different function from other layers so is done separately 
	CalculateOutputLayerBackwardsProp(networkLayers[networkLayers.size() - 1].get(), mLayerResults[mLayerResults.size() - 1].get(), correctAns);

	for (int i = (networkLayers.size() - 2); i > 0; i--)
		CalculateLayerBackwardsPropagation(networkLayers[i].get(), mLayerResults[i - 1].get());
}

void NeuralNetwork::CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* layerResults, int correctAns)
{
	if (!currentLayer || !currentLayer->GetPreviousLayer() || !layerResults)
	{
		throw std::invalid_argument("Null argument passed");
	}

	try
	{
		double y = 0.0;
		double tempTotalLoss = 0.0;

		std::vector<double> biasResults(currentLayer->GetLayerSize(), 0.0);
		std::vector<std::vector<double>> weightResults(currentLayer->GetLayerSize(), std::vector<double>(currentLayer->GetPreviousLayer()->GetLayerSize(), 0.0));

		for (int i = 0; i < currentLayer->GetLayerSize(); i++)
		{
			y = (correctAns == i) ? 1.0 : 0.0;
			const double activation = currentLayer->GetNeurons()[i]->GetActivation();
			const double deltaError = 2.0 * (activation - y);
			// Ensure that the activation function matches the derivative 
			const double deltaOutput = D_ActivationFunction(activation);

			tempTotalLoss += 0.5 * ((activation - y) * (activation - y)) * batchScale;

			currentLayer->GetNeurons()[i]->SetDeltaError(deltaError);
			currentLayer->GetNeurons()[i]->SetDeltaOutput(deltaOutput);
			biasResults[i] = deltaError * deltaOutput;

			for (int j = 0; j < currentLayer->GetPreviousLayer()->GetLayerSize(); j++)
			{
				const double prevLayerActivation = currentLayer->GetPreviousLayer()->GetNeurons()[j]->GetActivation();
				weightResults[i][j] += (deltaError * deltaOutput * prevLayerActivation) * batchScale;
			}
		}

		// If this point is reached no exceptions are thrown so update the layerResults
		layerResults->SetBiasResults(biasResults);
		layerResults->SetWeightResults(weightResults);
		totalLoss += tempTotalLoss;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw;
	}
}

void NeuralNetwork::CalculateLayerBackwardsPropagation(NetworkLayer* currentLayer, LayerResults* layerResults)
{
	if (!currentLayer || !currentLayer->GetPreviousLayer() || !currentLayer->GetNextLayer() || !layerResults)
	{
		throw std::invalid_argument("Null argument passed");
	}

	try
	{
		std::vector<double> biasResults(currentLayer->GetLayerSize(), 0.0);
		for (int i = 0; i < currentLayer->GetLayerSize(); i++)
		{
			const double activation = currentLayer->GetNeurons()[i]->GetActivation();
			// Ensure that the activation function matches the derivative 
			const double deltaOutput = D_ActivationFunction(activation);
			currentLayer->GetNeurons()[i]->SetDeltaError(0.0);
			currentLayer->GetNeurons()[i]->SetDeltaOutput(deltaOutput);


			for (int j = 0; j < currentLayer->GetNextLayer()->GetLayerSize(); j++)
			{
				const double deltaError = currentLayer->GetNeurons()[i]->GetDeltaError() +
					(
						currentLayer->GetNextLayer()->GetNeurons()[j]->GetDeltaError() *
						currentLayer->GetNextLayer()->GetNeurons()[j]->GetDeltaOutput() *
						currentLayer->GetNextLayer()->GetWeights()[j][i]
						);

				currentLayer->GetNeurons()[i]->SetDeltaError(deltaError);
			}

			biasResults[i] = currentLayer->GetNeurons()[i]->GetDeltaError() * currentLayer->GetNeurons()[i]->GetDeltaOutput();
		}

		std::vector<std::vector<double>> weightResults = layerResults->GetWeightResults();
		for (int i = 0; i < currentLayer->GetLayerSize(); i++)
		{
			for (int j = 0; j < currentLayer->GetPreviousLayer()->GetLayerSize(); j++)
			{
				weightResults[i][j] +=
					(
						currentLayer->GetNeurons()[i]->GetDeltaError() *
						currentLayer->GetNeurons()[i]->GetDeltaOutput() *
						currentLayer->GetPreviousLayer()->GetNeurons()[j]->GetActivation()
						);
			}
		}

		layerResults->SetBiasResults(biasResults);
		layerResults->SetWeightResults(weightResults);
	}
	catch (const std::exception&)
	{

	}

}

void NeuralNetwork::UpdateResults(int testSize)
{
	for (int i = 1; i < networkLayers.size(); i++)
		networkLayers[i]->UpdateBias(mLayerResults[i - 1].get(), learningRate * batchScale);

	for (int i = networkLayers.size() - 1; i > 0; i--)
		networkLayers[i]->UpdateWeight(mLayerResults[i - 1].get(), learningRate * batchScale);
}

void NeuralNetwork::BindActivationFunctions(ActivationFunctionTypes type)
{
	switch (type)
	{
	case ActivationFunctionTypes::Sigmoid:
		ActivationFunction = ActivationFunctions::Sigmoid;
		D_ActivationFunction = ActivationFunctions::D_Sigmoid;
		break;
	case ActivationFunctionTypes::ReLu:
		ActivationFunction = ActivationFunctions::ReLU;
		D_ActivationFunction = ActivationFunctions::D_ReLU;
		break;
	case ActivationFunctionTypes::LeakyReLu:
		ActivationFunction = ActivationFunctions::LeakyReLU;
		D_ActivationFunction = ActivationFunctions::D_Leaky_ReLU;
		break;
	default:
		throw std::runtime_error("Invalid activation function passed to network");
	}
}

std::vector<std::vector<double>> NeuralNetwork::InitalizeNetworkWeights(const int inputSize, const int outputSize)
{
	std::vector<std::vector<double>> weights(outputSize, std::vector<double>(inputSize));

	for (int i = 0; i < outputSize; i++)
	{
		for (int j = 0; j < inputSize; j++)
		{
			switch (activationFunctionType)
			{
			case ActivationFunctionTypes::Sigmoid:
				weights[i][j] = WeightInitializer::Xavier(inputSize, outputSize);
				break;
			case ActivationFunctionTypes::ReLu:
				weights[i][j] = WeightInitializer::He(inputSize);
				break;
			case ActivationFunctionTypes::LeakyReLu:
				weights[i][j] = WeightInitializer::He(inputSize);
				break;
			}
		}
	}
	return weights;
}

std::vector<double> NeuralNetwork::InitalizeBias(const std::size_t layerSize)
{
	double initalizedValue = 0.0;
	switch (activationFunctionType)
	{
	case ActivationFunctionTypes::Sigmoid:
		initalizedValue = 0.0;
		break;
	case ActivationFunctionTypes::ReLu:
		initalizedValue = 0.01;
		break;
	case ActivationFunctionTypes::LeakyReLu:
		initalizedValue = 0.0;
		break;
	}
	std::vector<double> bias(layerSize, initalizedValue);
	return bias;
}

void NeuralNetwork::ClipGradients(std::vector<std::vector<double>>& weights, double threshold)
{
	// Calculate L2 norm of the 2D gradient vector
	long double total_norm = 0.0;
	for (const auto& gradientRow : weights) {
		for (const double& gradientValue : gradientRow) {
			total_norm += gradientValue * gradientValue;
		}
	}
	total_norm = std::sqrt(total_norm);

	// If the total norm exceeds the threshold, clip the gradients
	if (total_norm > threshold) {
		double scale_factor = threshold / total_norm;
		for (auto& gradientRow : weights) {
			for (double& gradientValue : gradientRow) {
				gradientValue *= scale_factor;
			}
		}
	}
}

void NeuralNetwork::SaveWeightsAndBias(const std::string& filename) const
{
	NetworkLayer* layer = (networkLayers.size() > 1) ? networkLayers[1].get() : nullptr;
	try
	{
		if (!layer || !layer->GetPreviousLayer())
		{
			throw std::runtime_error("Null layer passed to SaveWeightsAndBias");
		}
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << '\n';
		throw;
	}

	FileManager file("Weights/" + filename, std::ios::out);

	while (layer)
	{
		for (int i = 0; i < layer->GetLayerSize(); i++)
		{
			for (int j = 0; j < layer->GetPreviousLayer()->GetLayerSize(); j++)
			{
				file.Write(reinterpret_cast<const char*>(&layer->GetWeights()[i][j]), sizeof(double));
			}
		}

		for (int i = 0; i < layer->GetLayerSize(); i++)
		{
			file.Write(reinterpret_cast<const char*>(&layer->GetBias()[i]), sizeof(double));
		}
		layer = layer->GetNextLayer().get();
	}

	DisplayManager::ClearConsole();
	std::cout << "Save Completed\n\n";
}

void NeuralNetwork::LoadWeightsAndBias(const std::string& filename) const
{
	NetworkLayer* layer = (networkLayers.size() > 1) ? networkLayers[1].get() : nullptr;
	try
	{
		if (!layer || !layer->GetPreviousLayer())
		{
			throw std::runtime_error("Null layer passed to SaveWeightsAndBias");
		}
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << '\n';
		throw;
	}

	FileManager file("Weights/" + filename, std::ios::in);
	NetworkLayer* layerIter = layer;
	
	// Using temporary variables to maintain strong exception safety
	while (layerIter)
	{
		std::vector<std::vector<double>> tempWeights = layerIter->GetWeights();
		std::vector<double> tempBias = layerIter->GetBias();

		for (int i = 0; i < layerIter->GetLayerSize(); i++)
		{
			for (int j = 0; j < layerIter->GetPreviousLayer()->GetLayerSize(); j++)
			{
				file.Read(reinterpret_cast<char*>(&tempWeights[i][j]), sizeof(double));
			}
		}

		for (int i = 0; i < layerIter->GetLayerSize(); i++)
		{
			file.Read(reinterpret_cast<char*>(&tempBias[i]), sizeof(double));
		}

		// If we reach here, it means no exceptions were thrown.
		layerIter->SetWeights(std::move(tempWeights));
		layerIter->SetBias(std::move(tempBias));

		layerIter = layerIter->GetNextLayer().get();
	}

	DisplayManager::ClearConsole();
	std::cout << "Weights loaded successfully\n\n";
}