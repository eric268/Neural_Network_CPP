#include "pch.h"
class Neurons;

class NetworkLayer
{
public:
	NetworkLayer();
	NetworkLayer(int numofNeurons);
	NetworkLayer(int numOfNeurons, NetworkLayer* prevLayer, NetworkLayer* nextLayer);
	NetworkLayer* mPreviousLayer;
	NetworkLayer* mNextLayer;
	int mNumberOfNeurons;
	std::vector<Neurons*> mNeurons;

};