#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/Connections.h"

#define InputLayerSize 784
#define HiddenLayer1Size 16
#define HiddenLayer2Size 16
#define OutputLayerSize 10

NeuralNetwork::NeuralNetwork() : mCostFound{0.0}, mDeltaCost {0.0}
{
	mInputLayer   = new NetworkLayer(LayerType::InputLayer, InputLayerSize);
	mHiddenLayer1 = new NetworkLayer(LayerType::HiddenLayer1, HiddenLayer1Size);
	mHiddenLayer2 = new NetworkLayer(LayerType::HiddenLayer2, HiddenLayer2Size);
	mOutputLayer  = new NetworkLayer(LayerType::OutputLayer, OutputLayerSize);

	mHiddenLayer1Results = LayerResults();
	mHiddenLayer2Results = LayerResults();
	mOutputLayerResults  = LayerResults();

	mInputLayer->mNextLayer = mHiddenLayer1;
	mHiddenLayer1->mPreviousLayer = mInputLayer;
	mHiddenLayer1->mNextLayer = mHiddenLayer2;
	mHiddenLayer2->mPreviousLayer = mHiddenLayer1;
	mHiddenLayer2->mNextLayer = mOutputLayer;
	mOutputLayer->mPreviousLayer = mHiddenLayer2;

	PopulateNeuronsInLayers(mInputLayer, mHiddenLayer1);
	PopulateNeuronsInLayers(mHiddenLayer1, mHiddenLayer2);
	PopulateNeuronsInLayers(mHiddenLayer2, mOutputLayer);

}

void NeuralNetwork::PopulateNeuronsInLayers(NetworkLayer* currentLayer, NetworkLayer* nextLayer)
{
	double low =  0.0;
	double high = 3.0;

	currentLayer->mWeights = std::vector<std::vector<double>>(currentLayer->mNumberOfNeurons, std::vector<double>(nextLayer->mNumberOfNeurons));
	std::random_device rd;
	std::uniform_int_distribution<int> dist(0, 500);

	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < nextLayer->mNumberOfNeurons; j++)
		{
			double val = ((double)dist(rd))/1000.0;
			currentLayer->mWeights[i][j] = val;
		}
	}

}

int NeuralNetwork::RunOneNumber(NetworkLayer* inputLayer, NetworkLayer* outputLayer, std::vector<double> pixelValues, int answer)
{
	for (int i = 0; i < pixelValues.size(); i++)
	{
		inputLayer->mNeurons[i]->mActivation = pixelValues[i]/255.0;
	}

	SetNextLayersActivation(mInputLayer, mHiddenLayer1);
	SetNextLayersActivation(mHiddenLayer1, mHiddenLayer2);
	SetNextLayersActivation(mHiddenLayer2, mOutputLayer);
	SetNextLayersActivation(mInputLayer, mOutputLayer);

	return GetFinalOutput(outputLayer);
}

int NeuralNetwork::GetFinalOutput(NetworkLayer* outputLayer)
{
	double highestActivation = -INFINITY;
	int i, ans = -1;
	for (i = 0; i < outputLayer->mNeurons.size(); i++)
	{
		if (highestActivation < outputLayer->mNeurons[i]->mActivation)
		{
			highestActivation = outputLayer->mNeurons[i]->mActivation;
			ans = i;
		}
	}
	return ans;
}

void NeuralNetwork::SetNextLayersActivation(NetworkLayer* currentLayer, NetworkLayer* nextLayer)
{
	double curr = 0.0;
	for (int next = 0; next < nextLayer->mNumberOfNeurons; next++)
	{
		nextLayer->mNeurons[next]->mActivation = 0.0;

		for (int current = 0; current < currentLayer->mNumberOfNeurons; current++)
		{
			nextLayer->mNeurons[next]->mActivation += currentLayer->mWeights[current][next] * currentLayer->mNeurons[current]->mActivation;
		}
		nextLayer->mNeurons[next]->mZ = nextLayer->mNeurons[next]->mActivation + nextLayer->mNeurons[next]->mBias;
		nextLayer->mNeurons[next]->mActivation = MathHelper::Sigmoid(nextLayer->mNeurons[next]->mZ);
	}
}

void NeuralNetwork::CalculateOutputLayerCost(int correctAns)
{
	mCostFound = 0.0;
	mDeltaCost = 0.0;
	for (int i = 0; i < mOutputLayer->mNumberOfNeurons; i++)
	{
		double y = 0.0;
		if (i == correctAns)
			y = 1.0;
		mCostFound += (mOutputLayer->mNeurons[i]->mActivation - y) * (mOutputLayer->mNeurons[i]->mActivation - y);
		mDeltaCost += 2.0 * (mOutputLayer->mNeurons[i]->mActivation - y);
	}

	mOutputLayerResults  += OutputLayerBackwardsInduction(mOutputLayer, mHiddenLayer2);
	mHiddenLayer2Results += OutputLayerBackwardsInduction(mHiddenLayer2, mHiddenLayer1);
	mHiddenLayer1Results += OutputLayerBackwardsInduction(mHiddenLayer1, mInputLayer);
}

LayerResults NeuralNetwork::OutputLayerBackwardsInduction(NetworkLayer* currentLayer, NetworkLayer* prevLayer)
{
	std::vector<double> mCurrentLayer2Bias(currentLayer->mNumberOfNeurons);
	std::vector<double> mPrevLayer2PrevActivation(prevLayer->mNumberOfNeurons);


	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < prevLayer->mNumberOfNeurons; i++)
		{
			mCurrentLayer2Bias[i] += MathHelper::DSigmoid(currentLayer->mNeurons[i]->mZ) * prevLayer->mNeurons[j]->mActivation;
		}
	}

	for (int i = 0; i < prevLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; i < currentLayer->mNumberOfNeurons; i++)
		{
			mPrevLayer2PrevActivation[i] += prevLayer->mWeights[j][i] * MathHelper::DSigmoid(currentLayer->mNeurons[i]->mZ) * mDeltaCost;
		}
	}
	std::vector<std::vector<double>> mPrevLayerWeightDelta(prevLayer->mNumberOfNeurons, std::vector<double>(currentLayer->mNumberOfNeurons));

	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < prevLayer->mNumberOfNeurons; j++)
		{
			mPrevLayerWeightDelta[j][i] = mPrevLayer2PrevActivation[j] * MathHelper::DSigmoid(currentLayer->mNeurons[i]->mZ) * mDeltaCost;
		}
	}

	return { mPrevLayerWeightDelta,mCurrentLayer2Bias };
}



