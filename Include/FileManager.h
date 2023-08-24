#pragma once
#include "../Include/pch.h"

class FileManager
{
public:

	FileManager(std::string filePath, std::ios::openmode mode) : file(std::fstream(filePath, std::ios::binary | mode))
	{
		if (!file.is_open())
			throw std::runtime_error("Failed to open file: " + filePath);
	}
	~FileManager()
	{
		file.close();
	}

	void Write(const char* data, std::streamsize size);
	void Read(char* data, std::streamsize size);

private:
	std::fstream file;
};