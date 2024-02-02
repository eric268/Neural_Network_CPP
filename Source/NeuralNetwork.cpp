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
	if (!layerSizes.size())
		throw std::invalid_argument("Empty network created");

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

int NeuralNetwork::RunNetwork(std::vector<double> pixelValues)
{
	// Input Layer
	SetNetworkInputs(pixelValues);

	// Hidden Layers
	for (int i = 1; i < networkLayers.size() - 1; i++)
		SetHiddenLayersActivation(networkLayers[i].get());

	//Output Layer
	SetOutputLayerActivation(networkLayers[networkLayers.size() - 1].get());

	// Returns the p
	return GetFinalOutput(networkLayers[networkLayers.size() - 1].get());
}

void NeuralNetwork::UpdateResults(int testSize)
{
	for (int i = 1; i < networkLayers.size(); i++)
		networkLayers[i]->UpdateBias(mLayerResults[i - 1].get(), learningRate * batchScale);

	for (int i = networkLayers.size() - 1; i > 0; i--)
		networkLayers[i]->UpdateWeight(mLayerResults[i - 1].get(), learningRate * batchScale);
}

void NeuralNetwork::ClearResults()
{
	totalLoss = 0.0;
	for (int i = 0; i < mLayerResults.size(); i++)
		mLayerResults[i]->ClearResults();
}

void NeuralNetwork::StartBackProp(int correctAns)
{
	//Output layer back prop is different function from other layers so is done separately 
	CalculateOutputLayerBackProp(networkLayers[networkLayers.size() - 1].get(), mLayerResults[mLayerResults.size() - 1].get(), correctAns);

	for (int i = (networkLayers.size() - 2); i > 0; i--)
		CalculateHiddenLayerBackProp(networkLayers[i].get(), mLayerResults[i - 1].get());
}

void NeuralNetwork::CalculateLoss(const int correctAns)
{
	if (networkLayers.size() < 2)
		throw std::invalid_argument("Invalid network size");

	std::shared_ptr<NetworkLayer> outputLayerPtr = networkLayers[networkLayers.size() - 1];
	for (int i = 0; i < outputLayerPtr->GetLayerSize(); i++)
	{
		totalLoss += GetOutputLoss(i, correctAns, outputLayerPtr->GetNeurons()[i]->GetActivation());
	}
}

void NeuralNetwork::SaveWeightsAndBias(const std::string& filename) const
{
	NetworkLayer* layer = (networkLayers.size() > 1) ? networkLayers[1].get() : nullptr;

	if (!layer || !layer->GetPreviousLayer())
		throw std::runtime_error("Null layer passed to SaveWeightsAndBias");

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
	if (!layer || !layer->GetPreviousLayer())
		throw std::runtime_error("Null layer passed to SaveWeightsAndBias");

	FileManager file("Weights/" + filename, std::ios::in);

	// Using temporary variables to maintain strong exception safety
	while (layer)
	{
		std::vector<std::vector<double>> tempWeights = layer->GetWeights();
		std::vector<double> tempBias = layer->GetBias();

		for (int i = 0; i < layer->GetLayerSize(); i++)
		{
			for (int j = 0; j < layer->GetPreviousLayer()->GetLayerSize(); j++)
			{
				file.Read(reinterpret_cast<char*>(&tempWeights[i][j]), sizeof(double));
			}
		}

		for (int i = 0; i < layer->GetLayerSize(); i++)
		{
			file.Read(reinterpret_cast<char*>(&tempBias[i]), sizeof(double));
		}

		// If we reach here, it means no exceptions were thrown.
		layer->SetWeights(std::move(tempWeights));
		layer->SetBias(std::move(tempBias));

		layer = layer->GetNextLayer().get();
	}

	DisplayManager::ClearConsole();
	std::cout << "Weights loaded successfully\n\n";
}

void NeuralNetwork::BindActivationFunctions(ActivationFunctionTypes type)
{
	switch (type)
	{
	case ActivationFunctionTypes::Sigmoid:
		ActivationFunction = ActivationFunctions::Sigmoid;
		D_ActivationFunction = ActivationFunctions::Sigmoid_Derivative;
		break;
	case ActivationFunctionTypes::ReLU:
		ActivationFunction = ActivationFunctions::ReLU;
		D_ActivationFunction = ActivationFunctions::ReLU_Derivative;
		break;
	case ActivationFunctionTypes::LeakyReLU:
		ActivationFunction = ActivationFunctions::LeakyReLU;
		D_ActivationFunction = ActivationFunctions::LeakyReLU_Derivative;
		break;
	default:
		throw std::runtime_error("Invalid activation function passed to network");
	}
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
			case ActivationFunctionTypes::ReLU:
				weights[i][j] = WeightInitializer::He(inputSize);
				break;
			case ActivationFunctionTypes::LeakyReLU:
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
	case ActivationFunctionTypes::ReLU:
		initalizedValue = 0.01;
		break;
	case ActivationFunctionTypes::LeakyReLU:
		initalizedValue = 0.0;
		break;
	}
	std::vector<double> bias(layerSize, initalizedValue);
	return bias;
}

void NeuralNetwork::SetNetworkInputs(std::vector<double> pixelValues)
{
	const size_t inputSize = pixelValues.size();
	assert(networkLayers.size() && networkLayers[0]->GetNeurons().size() == inputSize);

	std::vector<double> activation(inputSize, 0.0);
	for (int i = 0; i < inputSize; i++)
		activation[i] = pixelValues[i] / 255.0;

	// If this point is reached no exceptions were throw therefore update network layer
	for (int i = 0; i < inputSize; i++)
		networkLayers[0]->GetNeurons()[i]->SetActivation(activation[i]);
}

void NeuralNetwork::SetHiddenLayersActivation(NetworkLayer* currentLayer)
{
	if (!currentLayer || !currentLayer->GetPreviousLayer()) 
		throw std::invalid_argument("Invalid layer provided");


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

void NeuralNetwork::SetOutputLayerActivation(NetworkLayer* outputLayer)
{
	if (!outputLayer || !outputLayer->GetPreviousLayer()) 
		throw std::invalid_argument("Invalid layer provided");


	const size_t currentLayerSize = outputLayer->GetLayerSize();
	const size_t prevLayerSize = outputLayer->GetPreviousLayer()->GetLayerSize();

	// Local variable to store the new activations temporarily
	std::vector<double> newActivations(currentLayerSize, 0.0);

	// Calculate activations for the current layer
	for (int i = 0; i < currentLayerSize; i++) 
	{
		double sumActivation = 0.0;

		for (int j = 0; j < prevLayerSize; j++) 
		{
			sumActivation += outputLayer->GetWeights()[i][j] * outputLayer->GetPreviousLayer()->GetNeurons()[j]->GetActivation();
		}

		// Add the bias
		sumActivation += outputLayer->GetBias()[i];
		newActivations[i] = sumActivation;
	}

	// Calculate softmax probabilities
	std::vector<double> probabilities = ActivationFunctions::Softmax(newActivations);

	// If we reached here, all calculations were successful. Update the activations.
	for (int i = 0; i < currentLayerSize; i++) 
	{
		outputLayer->GetNeurons()[i]->SetActivation(probabilities[i]);
	}
}

int NeuralNetwork::GetFinalOutput(NetworkLayer* outputLayer)
{
	if (!outputLayer )
		throw std::invalid_argument("Invalid layer provided");

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

// Calculates the backward propagation for the output layer
void NeuralNetwork::CalculateOutputLayerBackProp(NetworkLayer* currentLayer, LayerResults* layerResults, int correctAns)
{
	// Check for null pointers and throw exceptions if necessary
	if (!currentLayer || !currentLayer->GetPreviousLayer() || !layerResults)
		throw std::invalid_argument("Null argument passed");

		// To store the temporary loss for this iteration
	double tempTotalLoss = 0.0;  

	// Initialize vectors to store various intermediate values
	std::vector<double> tempDeltaError(currentLayer->GetLayerSize(), 0.0);
	std::vector<double> tempDeltaOutput(currentLayer->GetLayerSize(), 0.0);
	std::vector<double> biasResults(layerResults->GetBiasResults());
	std::vector<std::vector<double>> weightResults(layerResults->GetWeightResults());

	// Loop through each neuron in the output layer
	for (int i = 0; i < currentLayer->GetLayerSize(); i++)
	{
		double activation = currentLayer->GetNeurons()[i]->GetActivation();

		// Set the correct label based on the index and correct answer
 		double y = (correctAns == i) ? 1.0 : 0.0;
 		// To avoid log(0)
 		const double epsilon = 1e-10;
 		// Calculate and accumulate the loss (cross-entropy)
 		tempTotalLoss += GetOutputLoss(i, correctAns, activation);

		// Calculate the derivative of the error for this neuron
		tempDeltaError[i] = activation - y;

		// Derivative of the output for an output layer that uses Soft Max and Categorical cross loss entropy is the same as the derivative of the error
		// Store this value so that it can be used when calculating back propagation for the hidden layers
		tempDeltaOutput[i] = tempDeltaError[i];
		biasResults[i] = tempDeltaError[i];

		for (int j = 0; j < currentLayer->GetPreviousLayer()->GetLayerSize(); j++)
		{
			const double prevLayerActivation = currentLayer->GetPreviousLayer()->GetNeurons()[j]->GetActivation();
			weightResults[i][j] += (tempDeltaError[i] * prevLayerActivation);
		}
	}

	// Update the layer with the new deltas and biases
	for (int i = 0; i < currentLayer->GetLayerSize(); i++)
	{
		currentLayer->GetNeurons()[i]->SetDeltaError(tempDeltaError[i]);
		currentLayer->GetNeurons()[i]->SetDeltaOutput(tempDeltaOutput[i]);
	}
	layerResults->SetBiasResults(biasResults);
	layerResults->SetWeightResults(weightResults);
	totalLoss += tempTotalLoss;  // Update the total loss
	
}

// Calculates the backward propagation for hidden layers
void NeuralNetwork::CalculateHiddenLayerBackProp(NetworkLayer* currentLayer, LayerResults* layerResults)
{
	// Check for null pointers and throw exceptions if necessary
	if (!currentLayer || !currentLayer->GetPreviousLayer() || !currentLayer->GetNextLayer() || !layerResults)
		throw std::invalid_argument("Null argument passed");

	// Initialize vectors to store various intermediate values
	std::vector<double> tempDeltaError(currentLayer->GetLayerSize(), 0.0);
	std::vector<double> tempDeltaOutput(currentLayer->GetLayerSize(), 0.0);

	std::vector<double> biasResults(layerResults->GetBiasResults());
	// Loop through each neuron in the hidden layer
	for (int i = 0; i < currentLayer->GetLayerSize(); i++)
	{
		// Get the activation of the neuron
		const double activation = currentLayer->GetNeurons()[i]->GetActivation();

		// Compute the derivative of the activation function
		tempDeltaOutput[i] = D_ActivationFunction(activation);

		// Compute the delta error for this neuron by summing the weighted delta errors from the next layer
		for (int j = 0; j < currentLayer->GetNextLayer()->GetLayerSize(); j++)
		{
			tempDeltaError[i] += currentLayer->GetNextLayer()->GetNeurons()[j]->GetDeltaError() *
				currentLayer->GetNextLayer()->GetWeights()[j][i];
		}

		// Multiply the sum by the derivative of the activation function to get the final delta error for this neuron
		tempDeltaError[i] *= tempDeltaOutput[i];

		// Update the bias based on the delta error
		biasResults[i] = tempDeltaError[i];
	}

	// Compute the weight updates for this layer
	std::vector<std::vector<double>> weightResults = layerResults->GetWeightResults();
	for (int i = 0; i < currentLayer->GetLayerSize(); i++)
	{
		for (int j = 0; j < currentLayer->GetPreviousLayer()->GetLayerSize(); j++)
		{
			weightResults[i][j] += tempDeltaError[i] *
				currentLayer->GetPreviousLayer()->GetNeurons()[j]->GetActivation();
		}
	}

	// Update the layer with the new deltas and biases
	for (int i = 0; i < currentLayer->GetLayerSize(); i++)
	{
		currentLayer->GetNeurons()[i]->SetDeltaError(tempDeltaError[i]);
		currentLayer->GetNeurons()[i]->SetDeltaOutput(tempDeltaOutput[i]);
	}
	layerResults->SetBiasResults(biasResults);
	layerResults->SetWeightResults(weightResults);
}

double NeuralNetwork::GetOutputLoss(const int layerIndex, const int correctAns, const double outputActivation)
{
	double y = (correctAns == layerIndex) ? 1.0 : 0.0;
	// To avoid log(0)
	const double epsilon = 1e-10;
	// Calculate and accumulate the loss (categorical cross-entropy)
	return -y * log(outputActivation + epsilon);
	
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