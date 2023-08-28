#pragma once

class Stopwatch
{
public:
	Stopwatch();
	~Stopwatch();

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;

	void Stop();
};