#pragma once

class Stopwatch
{
public:
	Stopwatch();
	~Stopwatch();

private:
	void Stop();
	std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;
};