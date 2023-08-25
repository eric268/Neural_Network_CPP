#pragma once

class Neurons
{
public:
	Neurons() : activation{ 0.0 }, deltaBias{ 0.0 }, deltaError{ 0.0 }, deltaOutput{ 0.0 } {}

#pragma region Inline Getters & Setters
	const double GetActivation()  const				{ return activation;  }
	const double GetDeltaBias()   const				{ return deltaBias;	  }
	const double GetDeltaError()  const				{ return deltaError;  }
	const double GetDeltaOutput() const				{ return deltaOutput; }

	void SetActivation (const double activation)	{ this->activation = activation; }
	void SetDeltaBias  (const double dBias)			{ deltaBias = dBias;			 }
	void SetDeltaError (const double dError)		{ deltaError = dError;			 }
	void SetDeltaOutput(const double dOutput)		{ deltaOutput = dOutput;		 }
#pragma endregion

private:
	double activation;
	double deltaBias;
	double deltaError;
	double deltaOutput;
};