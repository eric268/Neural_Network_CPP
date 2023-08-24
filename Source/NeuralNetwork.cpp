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
		networkLayers[i -1]->nextLayer = networkLayers[i];
		networkLayers[i]->previousLayer = networkLayers[i - 1];
		mLayerResults[i - 1] = std::make_shared<LayerResults>(layerSizes[i], layerSizes[i - 1]);
		PopulateNeuronsInLayers(networkLayers[i].get());
	}
	BindActivationFunctions(type);
}

void NeuralNetwork::PopulateNeuronsInLayers(NetworkLayer* currentLayer)
{
	if (!currentLayer || !currentLayer->previousLayer)
	{
		std::cerr << "Invalid layer or previous layer is nullptr.\n";
		throw std::invalid_argument("Invalid layer provided");
	}

	const int currentLayerSize = currentLayer->numberOfNeurons;
	const int prevLayerSize = currentLayer->previousLayer->numberOfNeurons;

	currentLayer->weights = std::vector<std::vector<double>>(currentLayerSize, std::vector<double>(prevLayerSize));
	InitalizeNetworkWeights(currentLayer->weights, prevLayerSize, currentLayerSize);

	currentLayer->bias = std::vector<double>(currentLayerSize);
	InitalizeBias(currentLayer->bias);
}


void NeuralNetwork::ClearResults()
{
	totalLoss = 0.0;
	double mTotalLossCopy = totalLoss;
	auto mLayerResultCopy = mLayerResults;

	try
	{
		for (int i = 0; i < mLayerResultCopy.size(); i++)
		{
			std::fill(mLayerResultCopy[i]->biasResults.begin(), mLayerResultCopy[i]->biasResults.end(), 0.0);

			for (auto& p : mLayerResultCopy[i]->weightedResults)
			{
				std::fill(p.begin(), p.end(), 0.0);
			}
		}

		mLayerResults.swap(mLayerResultCopy);
		totalLoss = mTotalLossCopy;
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
	assert(networkLayers.size() && networkLayers[0]->neurons.size() == pixelValues.size());

	const size_t inputSize = pixelValues.size();
	try
	{
		std::vector<double> activation (inputSize, 0.0);
		for (int i = 0; i < inputSize; i++)
			activation[i] = pixelValues[i] / 255.0;

		// If this point is reached no exceptions were throw therefore update network layer
		for (int i =0; i < inputSize; i++)
			networkLayers[0]->neurons[i]->SetActivation(activation[i]);
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw;
	}
}

void NeuralNetwork::SetHiddenLayersActivation(NetworkLayer* currentLayer)
{
	if (!currentLayer || !currentLayer->previousLayer) {
		std::cerr << "Invalid layer or previous layer is nullptr.\n";
		throw std::invalid_argument("Invalid layer provided");
	}

	try
	{
		const int currentLayerSize = currentLayer->numberOfNeurons;
		const int prevLayerSize = currentLayer->previousLayer->numberOfNeurons;

		std::vector<double> newActivations(currentLayerSize, 0.0);
		for (int i = 0; i < currentLayerSize; i++)
		{
			double sumActivation = 0.0;
			for (int j = 0; j < prevLayerSize; j++)
			{
				sumActivation += currentLayer->weights[i][j] * currentLayer->previousLayer->neurons[j]->GetActivation();
			}

			// Add bias and apply activation function
			newActivations[i] = ActivationFunction(sumActivation + currentLayer->bias[i]);
		}

		// If this point is reached no exceptions were throw therefore update network layer
		for (int i = 0; i < currentLayerSize; i++) 
		{
			currentLayer->neurons[i]->SetActivation(newActivations[i]);
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
	if (!outputLayer || !outputLayer->previousLayer) 
	{
		std::cerr << "Invalid layer or previous layer is nullptr.\n";
		throw std::invalid_argument("Invalid layer provided");
	}

	try 
	{
		const int currentLayerSize = outputLayer->numberOfNeurons;
		const int prevLayerSize = outputLayer->previousLayer->numberOfNeurons;

		// Local variable to store the new activations temporarily
		std::vector<double> newActivations(currentLayerSize, 0.0);

		// Calculate activations for the current layer
		for (int i = 0; i < currentLayerSize; ++i) {
			double sumActivation = 0.0;

			for (int j = 0; j < prevLayerSize; ++j) {
				sumActivation += outputLayer->weights[i][j] * outputLayer->previousLayer->neurons[j]->GetActivation();
			}

			// Add the bias
			sumActivation += outputLayer->bias[i];
			newActivations[i] = sumActivation;
		}

		// Calculate softmax probabilities
		std::vector<double> probabilities = ActivationFunctions::softmax(newActivations);

		// If we reached here, all calculations were successful. Update the activations.
		for (int i = 0; i < currentLayerSize; ++i) {
			outputLayer->neurons[i]->SetActivation(probabilities[i]);
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
	for (i = 0; i < outputLayer->neurons.size(); i++)
	{
		if (highestActivation < outputLayer->neurons[i]->GetActivation())
		{
			highestActivation = outputLayer->neurons[i]->GetActivation();
			ans = i;
		}
	}
	return ans;
}


void NeuralNetwork::CalculateLayerDeltaCost(int correctAns)
{
	//Output layer back prop is different function from other layers so is done separately 
	CalculateOutputLayerBackwardsProp(networkLayers[networkLayers.size() - 1].get(), mLayerResults[mLayerResults.size() - 1].get(), correctAns);

	for (int i = static_cast<int>((networkLayers.size() - 2)); i > 0; i--)
		CalculateLayerBackwardsPropagation(networkLayers[i].get(), mLayerResults[i - 1].get());
}

void NeuralNetwork::CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, LayerResults* layerResults, int correctAns)
{
	if (!currentLayer || !currentLayer->previousLayer || !layerResults)
	{
		throw std::invalid_argument("Null argument passed");
	}

	double y = 0.0;
	double tempTotalLoss = 0.0;

	for (int i = 0; i < currentLayer->numberOfNeurons; i++)
	{
		y = (correctAns == i) ? 1.0 : 0.0;
		const double activation = currentLayer->neurons[i]->GetActivation();
		const double deltaError = 2.0 * (activation - y);
		// Ensure that the activation function matches the derivative 
		const double deltaOutput = D_ActivationFunction(activation);

		tempTotalLoss += 0.5 * ((activation - y) * (activation - y)) * batchScale;

		currentLayer->neurons[i]->SetDeltaError(deltaError);
		currentLayer->neurons[i]->SetDeltaOutput(deltaOutput);
		layerResults->biasResults[i] = deltaError * deltaOutput;

		for (int j = 0; j < currentLayer->previousLayer->numberOfNeurons; j++)
		{
			const double prevLayerActivation = currentLayer->previousLayer->neurons[j]->GetActivation();
			layerResults->weightedResults[i][j] += (deltaError * deltaOutput * prevLayerActivation) * batchScale;
		}
	}
	totalLoss += tempTotalLoss;
}

void NeuralNetwork::CalculateLayerBackwardsPropagation(NetworkLayer* currentLayer, LayerResults* layerResults)
{
	if (!currentLayer || !currentLayer->previousLayer || !layerResults)
	{
		throw std::invalid_argument("Null argument passed");
	}

	for (int i = 0; i < currentLayer->numberOfNeurons; i++)
	{
		const double activation = currentLayer->neurons[i]->GetActivation();
		// Ensure that the activation function matches the derivative 
		const double deltaOutput = D_ActivationFunction(activation);
		currentLayer->neurons[i]->SetDeltaError(0.0);
		currentLayer->neurons[i]->SetDeltaOutput(deltaOutput);


		for (int j = 0; j < currentLayer->nextLayer->numberOfNeurons; j++)
		{
			const double deltaError = currentLayer->neurons[i]->GetDeltaError()	+ 
				(
					currentLayer->nextLayer->neurons[j]->GetDeltaError()			* 
					currentLayer->nextLayer->neurons[j]->GetDeltaOutput()			*
					currentLayer->nextLayer->weights[j][i]
				);

			currentLayer->neurons[i]->SetDeltaError(deltaError);
		}

		layerResults->biasResults[i] = currentLayer->neurons[i]->GetDeltaError() * currentLayer->neurons[i]->GetDeltaOutput();
	}

	for (int i = 0; i < currentLayer->numberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->previousLayer->numberOfNeurons; j++)
		{
			layerResults->weightedResults[i][j] += 
			(
				currentLayer->neurons[i]->GetDeltaError()					* 
				currentLayer->neurons[i]->GetDeltaOutput()					* 
				currentLayer->previousLayer->neurons[j]->GetActivation()
			);
		}
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

void NeuralNetwork::InitalizeNetworkWeights(std::vector<std::vector<double>>& weights, const int inputSize, const int outputSize)
{
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
}

void NeuralNetwork::InitalizeBias(std::vector<double>& bias)
{
	for (int i =0; i < bias.size(); i++)
	{
		switch (activationFunctionType)
		{
		case ActivationFunctionTypes::Sigmoid:
			bias[i] = 0.0;
			break;
		case ActivationFunctionTypes::ReLu:
			bias[i] = 0.01;
			break;
		case ActivationFunctionTypes::LeakyReLu:
			bias[i] = 0.0;
			break;
		}
	}
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
		if (!layer || !layer->previousLayer)
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
		for (int i = 0; i < layer->numberOfNeurons; i++)
		{
			for (int j = 0; j < layer->previousLayer->numberOfNeurons; j++)
			{
				file.Write(reinterpret_cast<const char*>(&layer->weights[i][j]), sizeof(double));
			}
		}

		for (int i = 0; i < layer->numberOfNeurons; i++)
		{
			file.Write(reinterpret_cast<const char*>(&layer->bias[i]), sizeof(double));
		}
		layer = layer->nextLayer.get();
	}

	DisplayManager::ClearConsole();
	std::cout << "Save Completed\n\n";
}

void NeuralNetwork::LoadWeightsAndBias(const std::string& filename) const
{
	NetworkLayer* layer = (networkLayers.size() > 1) ? networkLayers[1].get() : nullptr;
	try
	{
		if (!layer || !layer->previousLayer)
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
		std::vector<std::vector<double>> tempWeights = layerIter->weights;
		std::vector<double> tempBias = layerIter->bias;

		for (int i = 0; i < layerIter->numberOfNeurons; i++)
		{
			for (int j = 0; j < layerIter->previousLayer->numberOfNeurons; j++)
			{
				file.Read(reinterpret_cast<char*>(&tempWeights[i][j]), sizeof(double));
			}
		}

		for (int i = 0; i < layerIter->numberOfNeurons; i++)
		{
			file.Read(reinterpret_cast<char*>(&tempBias[i]), sizeof(double));
		}

		// If we reach here, it means no exceptions were thrown.
		layerIter->weights = std::move(tempWeights);
		layerIter->bias = std::move(tempBias);

		layerIter = layerIter->nextLayer.get();
	}

	DisplayManager::ClearConsole();
	std::cout << "Weights loaded successfully\n\n";
}