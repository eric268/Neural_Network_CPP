#pragma once

class WeightInitializer
{
public:
	static double Xavier(const int inputSize, const int outputSize)
	{
		double variance = 2.0 / (static_cast<double>(inputSize) + static_cast<double>(outputSize));
		return SampleNormal(0, std::sqrt(variance));
	}

	static double He(int inputSize)
	{
		double variance = 2.0 / static_cast<double>(inputSize);
		return SampleNormal(0, std::sqrt(variance));
	}

private:
	static double SampleNormal(double mean, double stddev)
	{
		static std::mt19937 generator(std::random_device{}());
		std::normal_distribution<double> dist(mean, stddev);
		return dist(generator);
	}
};