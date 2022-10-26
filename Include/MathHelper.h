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
	//template<typename WorksForSigmoid>
	//static double FastSigmoid(WorksForSigmoid x)
	//{
	//	return (x / (1.0 + abs(x)));
	//}
	//template<typename WorksForSigmoid>
	//static double DFastSigmoid(WorksForSigmoid x)
	//{
	//	double val = FastSigmoid(x);
	//	return (val * (1.0 - val));
	//}

	template<typename WorksForSigmoid>
	static double Sigmoid(WorksForSigmoid x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	template <typename WorksForSigmoid>
	static double DSigmoid(WorksForSigmoid x)
	{
		double val = Sigmoid(x);
		return (val * (1.0 - val));
	}
};



