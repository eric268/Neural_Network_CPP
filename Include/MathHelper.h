#pragma once
#include "pch.h"

template<typename T>
concept WorksForSigmoid = requires(T a)
{
	{a + 1.0};
	{abs(a)};
	{a / 1.0};
	std::is_convertible_v<T,double>;
};

class MathHelper
{
public:
	template<typename WorksForSigmoid>
	static double Sigmoid(const WorksForSigmoid x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	static double RELUI(double x)
	{
		return std::max(0.0, x);
	}

	template <typename WorksForSigmoid>
	static double DSigmoid(const WorksForSigmoid x)
	{
		double val = Sigmoid(x);
		return (val * (1.0 - val));
	}

	template <typename WorksForSigmoid>
	static double DReLU(const WorksForSigmoid x)
	{
		return (x <= 0.0) ? 0.0 : 1.0;
	}
};



