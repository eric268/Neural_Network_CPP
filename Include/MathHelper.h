﻿#pragma once
#include "pch.h"

class MathHelper
{
public:

	static double Sigmoid(const double x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	// This assumes that Sigmoid(x) has already been completed
	static double D_Sigmoid(const double x)
	{
		return (x * (1.0 - x));
	}

	static double ReLu(double x)
	{
		return std::max(0.0, x);
	}

	static double D_ReLU(const double x)
	{
		return (x <= 0.0) ? 0.0 : 1.0;
	}

	static double LeakyReLu(double x, double alpha)
	{
		return (x > 0) ? x : (alpha * x);
	}

	static double D_Leaky_ReLu(double x, double alpha)
	{
		return (x > 0) ? 1.0 : alpha;
	}

	static std::vector<double> softmax(const std::vector<double>& logits) 
	{
		std::vector<double> probabilities;
		double maxLogit = *std::max_element(logits.begin(), logits.end());
		double sumExp = 0.0;

		for (double logit : logits) {
			double expLogit = std::exp(logit - maxLogit);
			probabilities.push_back(expLogit);
			sumExp += expLogit;
		}

		for (double& prob : probabilities) {
			prob /= sumExp;
		}

		return probabilities;
	}
};



