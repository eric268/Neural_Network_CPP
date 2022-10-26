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
	static double FastSigmoid(double x)
	{
		return x / (1.0 + abs(x));
	}
};



