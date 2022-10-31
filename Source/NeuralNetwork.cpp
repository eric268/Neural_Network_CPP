#include "../Include/pch.h"
#include "../Include/NeuralNetwork.h"
#include "../Include/NetworkLayer.h"
#include "../Include/Neurons.h"
#include "../Include/Connections.h"

NeuralNetwork::NeuralNetwork() : mTotalError {0.0}
{
	mInputLayer   = new NetworkLayer(LayerType::InputLayer, InputLayerSize);
	mHiddenLayer1 = new NetworkLayer(LayerType::HiddenLayer1, HiddenLayer1Size);
	mHiddenLayer2 = new NetworkLayer(LayerType::HiddenLayer2, HiddenLayer2Size);
	mOutputLayer  = new NetworkLayer(LayerType::OutputLayer, OutputLayerSize);

	mHiddenLayer1Results = LayerResults(HiddenLayer1Size, InputLayerSize);
	mHiddenLayer2Results = LayerResults(HiddenLayer2Size, HiddenLayer1Size);
	mOutputLayerResults  = LayerResults(OutputLayerSize, HiddenLayer2Size);

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
	currentLayer->mWeights = std::vector<std::vector<double>>(nextLayer->mNumberOfNeurons, std::vector<double>(currentLayer->mNumberOfNeurons));
	std::random_device rd;
	std::uniform_int_distribution<int> dist(-10000, 10000);

	for (int i = 0; i < nextLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->mNumberOfNeurons; j++)
		{
			double val = ((double)dist(rd))*0.00001;
			currentLayer->mWeights[i][j] = val;
		}
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

	return GetFinalOutput(mOutputLayer);
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
	for (int i = 0; i < nextLayer->mNumberOfNeurons; i++)
	{
		nextLayer->mNeurons[i]->mActivation = 0.0;
		for (int j = 0; j < currentLayer->mNumberOfNeurons; j++)
		{
			nextLayer->mNeurons[i]->mActivation += currentLayer->mWeights[i][j] * currentLayer->mNeurons[j]->mActivation;
		}
		nextLayer->mNeurons[i]->mActivation = MathHelper::Sigmoid(nextLayer->mNeurons[i]->mActivation + nextLayer->mNeurons[i]->mBias);
	}
}

void NeuralNetwork::CalculateLayerDeltaCost(int correctAns)
{
#pragma region Debug
	//NetworkLayer* i = new NetworkLayer(LayerType::InputLayer, 2);
	//NetworkLayer* h = new NetworkLayer(LayerType::HiddenLayer1, 2);
	//NetworkLayer* o = new NetworkLayer(LayerType::OutputLayer, 2);

	//i->mNextLayer = h;
	//h->mNextLayer = o;
	//h->mPreviousLayer = i;
	//o->mPreviousLayer = h;

	//i->mWeights = std::vector < std::vector<double>>(2, std::vector<double>(2));
	//i->mWeights[0] = { 0.15, 0.2 };
	//i->mWeights[1] = { 0.25, 0.3 };
	//i->mNeurons[0]->mActivation = 0.05;
	//i->mNeurons[1]->mActivation = 0.1;

	//h->mWeights = std::vector < std::vector<double>>(2, std::vector<double>(2));
	//h->mWeights[0] = { 0.4, 0.45 };
	//h->mWeights[1] = { 0.5, 0.55 };
	//h->mNeurons[0]->mBias = 0.35;
	//h->mNeurons[1]->mBias = 0.35;

	//o->mNeurons[0]->mBias = 0.6;
	//o->mNeurons[1]->mBias = 0.6;

	//SetNextLayersActivation(i, h);
	//SetNextLayersActivation(h, o);
	//auto result = CalculateOutputLayerBackwardsProp(o, h, 0);
	//auto result2 = CalculateLayerBackwardsPropigation(h, 0);
#pragma endregion

	mOutputLayerResults  = mOutputLayerResults  + CalculateOutputLayerBackwardsProp(mOutputLayer, correctAns);
	mHiddenLayer2Results = mHiddenLayer2Results + CalculateLayerBackwardsPropigation(mHiddenLayer2, correctAns);
	mHiddenLayer1Results = mHiddenLayer1Results + CalculateLayerBackwardsPropigation(mHiddenLayer1, correctAns);
}

LayerResults NeuralNetwork::CalculateOutputLayerBackwardsProp(NetworkLayer* currentLayer, int correctAns)
{
	std::vector<double> mCurrentLayer2Bias(currentLayer->mNumberOfNeurons, 0.0);
	std::vector<std::vector<double>> mPrevLayerWeightDelta(currentLayer->mNumberOfNeurons, std::vector<double>(currentLayer->mPreviousLayer->mNumberOfNeurons, 0.0));

	double y = 0.0;

	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		//y = (correctAns == i) ? 0.01 : 0.99;
		y = (correctAns == i) ? 1.0 : 0.01;
		mTotalError += (currentLayer->mNeurons[i]->mActivation - y) * (currentLayer->mNeurons[i]->mActivation - y);
		//currentLayer->mNeurons[i]->mDeltaError = -1.0 * (y - currentLayer->mNeurons[i]->mActivation);
		currentLayer->mNeurons[i]->mDeltaError = 2.0 * (currentLayer->mNeurons[i]->mActivation - y);
		currentLayer->mNeurons[i]->mDeltaOutput = currentLayer->mNeurons[i]->mActivation * (1.0 - currentLayer->mNeurons[i]->mActivation);
		mCurrentLayer2Bias[i] = currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput;


		for (int j = 0; j < currentLayer->mPreviousLayer->mNumberOfNeurons; j++)
		{
			//mPrevLayerWeightDelta[i][j] = prevLayer->mWeights[i][j] - 0.5 * currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput * prevLayer->mNeurons[j]->mActivation;
			mPrevLayerWeightDelta[i][j] = currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput * currentLayer->mPreviousLayer->mNeurons[j]->mActivation;
		}
	}

	return { mPrevLayerWeightDelta,mCurrentLayer2Bias };
}

LayerResults NeuralNetwork::CalculateLayerBackwardsPropigation(NetworkLayer* currentLayer, int correctAns)
{
	std::vector<std::vector<double>> weights = std::vector<std::vector<double>>(currentLayer->mNumberOfNeurons, std::vector<double>(currentLayer->mPreviousLayer->mNumberOfNeurons));
	std::vector<double> bias(currentLayer->mNumberOfNeurons, 0.0);
	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		currentLayer->mNeurons[i]->mDeltaError = 0.0;
		currentLayer->mNeurons[i]->mDeltaOutput = currentLayer->mNeurons[i]->mActivation * (1.0 - currentLayer->mNeurons[i]->mActivation);
		for (int j = 0; j < currentLayer->mNextLayer->mNumberOfNeurons; j++)
		{
			currentLayer->mNeurons[i]->mDeltaError += currentLayer->mNextLayer->mNeurons[j]->mDeltaError 
				* currentLayer->mNextLayer->mNeurons[j]->mDeltaOutput * currentLayer->mWeights[j][i];
		}
		bias[i] = currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput;
	}

	for (int i = 0; i < currentLayer->mNumberOfNeurons; i++)
	{
		for (int j = 0; j < currentLayer->mPreviousLayer->mNumberOfNeurons; j++)
		{
			//weights[i][j] = currentLayer->mPreviousLayer->mWeights[i][j] - 0.5 * val;
			weights[i][j] = currentLayer->mNeurons[i]->mDeltaError * currentLayer->mNeurons[i]->mDeltaOutput * currentLayer->mPreviousLayer->mNeurons[j]->mActivation;
		}
	}

	return { weights,bias };
}


