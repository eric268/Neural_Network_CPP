#pragma once

class Neurons
{
public:
	Neurons();
	~Neurons() = default;

#pragma region Inline Functions
	inline const double GetActivation() const				{ return activation;  }
	inline const double GetDeltaBias() const				{ return deltaBias;	  }
	inline const double GetDeltaError() const				{ return deltaError;  }
	inline const double GetDeltaOutput() const				{ return deltaOutput; }

	inline void SetActivation (const double activation)		{ this->activation = activation; }
	inline void SetDeltaBias (const double dBias)			{ deltaBias = dBias;			 }
	inline void SetDeltaError (const double dError)			{ deltaError = dError;			 }
	inline void SetDeltaOutput(const double dOutput)		{ deltaOutput = dOutput;		 }
#pragma endregion

private:
	double activation;
	double deltaBias;
	double deltaError;
	double deltaOutput;
};