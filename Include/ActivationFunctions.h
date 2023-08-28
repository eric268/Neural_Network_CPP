#pragma once

class ActivationFunctions
{
public:

	static double Sigmoid(const double x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	// This assumes that Sigmoid(x) has already been executed on input parameter via forward pass
	static double Sigmoid_Derivative(const double x)
	{
		return (x * (1.0 - x));
	}

	static double ReLU(const double x)
	{
		return std::max(0.0, x);
	}

	static double ReLU_Derivative(const double x)
	{
		return (x <= 0.0) ? 0.0 : 1.0;
	}

	static double LeakyReLU(const double x)
	{
		const double alpha = 0.001;
		return (x > 0) ? x : (alpha * x);
	}

	static double LeakyReLU_Derivative(double x)
	{
		const double alpha = 0.001;
		return (x >= 0) ? 1.0 : alpha;
	}

	static std::vector<double> Softmax(const std::vector<double>& logits) 
	{
		const size_t size = logits.size();
		std::vector<double> probabilities(size);
		const double maxLogit = *std::max_element(logits.begin(), logits.end());
		double sumExp = 0.0;

		for (int i = 0; i < size; i++) 
		{
			double expLogit = std::exp(logits[i] - maxLogit);
			probabilities[i] = expLogit;
			sumExp += expLogit;
		}

		for (double& prob : probabilities) 
		{
			prob /= sumExp;
		}

		return probabilities;
	}
};



