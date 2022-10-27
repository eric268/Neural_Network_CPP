#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/Connections.h"

NeuralNetwork::NeuralNetwork()
{
	mInputLayer   = new NetworkLayer(784);
	mHiddenLayer1 = new NetworkLayer(16);
	mHiddenLayer2 = new NetworkLayer(16);
	mOutputLayer  = new NetworkLayer(10);

	PopulateNeuronsInLayers(mInputLayer, mHiddenLayer1);
	PopulateNeuronsInLayers(mHiddenLayer1, mHiddenLayer2);
	PopulateNeuronsInLayers(mHiddenLayer2, mOutputLayer);
}

void NeuralNetwork::PopulateNeuronsInLayers(NetworkLayer* currentLayer, NetworkLayer* nextLayer)
{
	for (int i = 0; i < currentLayer->mNeurons.size(); i++)
	{
		currentLayer->mNeurons[i]->PopulateConnections(nextLayer);
	}
}

int NeuralNetwork::RunOneNumber(std::vector<double> pixelValues, int answer)
{
	for (int i = 0; i < pixelValues.size(); i++)
	{
		mInputLayer->mNeurons[i]->mActivation = pixelValues[i]/255.0;
	}

	SetNextLayersActivation(mInputLayer, mHiddenLayer1);
	SetNextLayersActivation(mHiddenLayer1, mHiddenLayer2);
	SetNextLayersActivation(mHiddenLayer2, mOutputLayer);

	double highestActivation = -INFINITY;
	int i, ans;

	for (i = 0; i < mOutputLayer->mNeurons.size(); i++)
	{
		if (highestActivation < mOutputLayer->mNeurons[i]->mActivation)
		{
			highestActivation = mOutputLayer->mNeurons[i]->mActivation;
			ans = i;
		}
	}

	return ans;
}

void NeuralNetwork::SetNextLayersActivation(NetworkLayer* currentLayer, NetworkLayer* nextLayer)
{
	for (int i = 0; i < nextLayer->mNeurons.size(); i++)
	{
		nextLayer->mNeurons[i]->mActivation = 0.0;
		for (int j = 0; j < currentLayer->mNeurons.size(); j++)
		{
			nextLayer->mNeurons[i]->mActivation += (currentLayer->mNeurons[j]->mConnections[i]->mWeight * currentLayer->mNeurons[j]->mActivation);
		}
		nextLayer->mNeurons[i]->mActivation = MathHelper::Sigmoid(nextLayer->mNeurons[i]->mActivation + nextLayer->mNeurons[i]->mBias);
	}
}

void NeuralNetwork::SerializeWeights()
{
	std::ofstream file("Weight.dat", std::ios::out | std::ios::binary);
	if (file.is_open())
	{

	}
	else
	{
		std::cerr << "Failed to open Weights.dat file\n";
	}
}

void NeuralNetwork::SerializeBias()
{
	std::ofstream file("BiasValues.dat", std::ios::out | std::ios::binary);
	if (file.is_open())
	{

	}
	else
	{
		std::cerr << "Failed to open Weights.dat file\n";
	}
}

void NeuralNetwork::CalculateCosts(int correctAns)
{

}
