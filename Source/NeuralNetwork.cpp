#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/Connections.h"

NeuralNetwork::NeuralNetwork() : mCostFound{0.0}, mDeltaCost {0.0}
{
	mInputLayer   = new NetworkLayer(LayerType::InputLayer, InputLayerSize);
	mHiddenLayer1 = new NetworkLayer(LayerType::HiddenLayer1, HiddenLayer1Size);
	mHiddenLayer2 = new NetworkLayer(LayerType::HiddenLayer2, HiddenLayer2Size);
	mOutputLayer  = new NetworkLayer(LayerType::OutputLayer, OutputLayerSize);

	mHiddenLayer1Results = LayerResults(InputLayerSize, HiddenLayer1Size);
	mHiddenLayer2Results = LayerResults(HiddenLayer1Size, HiddenLayer2Size);
	mOutputLayerResults  = LayerResults(HiddenLayer2Size, OutputLayerSize);

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
	currentLayer->mWeights = std::vector<std::vector<double>>(currentLayer->mNumberOfNeurons, std::vector<double>(nextLayer->mNumberOfNeurons));
	std::random_device rd;
	std::uniform_int_distribution<int> dist(-10000, 10000);

	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < nextLayer->mNumberOfNeurons; j++)
		{
			double val = ((double)dist(rd))*0.00001;
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

void NeuralNetwork::CalculateLayerDeltaCost(int correctAns)
{
	mCostFound = 0.0;
	mDeltaCost = 0.0;
	for (int i = 0; i < mOutputLayer->mNumberOfNeurons; i++)
	{
		double y = 0.0;
		if (i == correctAns)
			y = 1.0;
		double val = mOutputLayer->mNeurons[i]->mActivation - y;
		mCostFound += val * val;
		mDeltaCost += 2.0 * val;
	}

	mOutputLayerResults  = mOutputLayerResults  + CalculateLayerBackwardsPropigation(mOutputLayer, mHiddenLayer2);
	mHiddenLayer2Results = mHiddenLayer2Results + CalculateLayerBackwardsPropigation(mHiddenLayer2, mHiddenLayer1);
	mHiddenLayer1Results = mHiddenLayer1Results + CalculateLayerBackwardsPropigation(mHiddenLayer1, mInputLayer);
}

LayerResults NeuralNetwork::CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, NetworkLayer* prevLayer)
{
	std::vector<double> mCurrentLayer2Bias(currentLayer->mNumberOfNeurons, 0.0);
	std::vector<double> mPrevLayer2PrevActivation(prevLayer->mNumberOfNeurons, 0.0);
	std::vector<std::vector<double>> mPrevLayerWeightDelta(prevLayer->mNumberOfNeurons, std::vector<double>(currentLayer->mNumberOfNeurons, 0.0));

	bool doOnce = false;

	for (int i = 0; i < prevLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->mNumberOfNeurons; j++)
		{
			mPrevLayer2PrevActivation[i] += prevLayer->mWeights[i][j] * MathHelper::DSigmoid(currentLayer->mNeurons[j]->mZ) * mDeltaCost;
			if (!doOnce)
			{
				mCurrentLayer2Bias[j] = MathHelper::DSigmoid(currentLayer->mNeurons[j]->mZ) * mDeltaCost;
			}
		}
		doOnce = true;
	}


	for (int i = 0; i < prevLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->mNumberOfNeurons; j++)
		{
			mPrevLayerWeightDelta[i][j] = mPrevLayer2PrevActivation[i] * MathHelper::DSigmoid(currentLayer->mNeurons[j]->mZ) * mDeltaCost;
		}
	}

	return { mPrevLayerWeightDelta,mCurrentLayer2Bias };
}



