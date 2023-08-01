#pragma once
#include "pch.h"

template<typename T>
concept WorksForSigmoid = requires(T a)
{
	{a + 1.0};
	{abs(a)};
	{a / 1.0};
	std::is_convertible_v<T,float>;
};

class MathHelper
{
public:
	template<typename WorksForSigmoid>
	static float Sigmoid(const WorksForSigmoid x)
	{
		return (1.0 / (1.0 + exp(-x)));
	}

	static float ReLu(float x)
	{
		return std::max(0.0f, x);
	}

	template <typename WorksForSigmoid>
	static float DSigmoid(const WorksForSigmoid x)
	{
		float val = Sigmoid(x);
		return (val * (1.0 - val));
	}

	template <typename WorksForSigmoid>
	static float DReLU(const WorksForSigmoid x)
	{
		return (x <= 0.0) ? 0.0 : 1.0;
	}
};



