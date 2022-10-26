#pragma once
#include "pch.h"

class Connections;
class NetworkLayer;
class Neurons
{
public:
	Neurons();
	Neurons(std::vector<double> imageInput);
	Neurons(std::vector<Neurons*> prevNerons);

	void AddConnections(std::vector<Connections*> nextLayerOfNeurons);


	double CalculateCost(bool isCorrect);
	double DCalculateCost(bool isCorrect);

	void UpdateNeuron();

	void PopulateConnections(NetworkLayer& nextLayer);

public:
	std::vector<Connections*> mConnections;
	double mBias;
	double mActivation;
	double mZ;
};

/*what do I need for backwards propigations
 - y (done)
 - weight and activation are different!

*/