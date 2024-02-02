#include "../Include/pch.h"
#include "../Include/FileManager.h"

FileManager::FileManager(std::string filePath, std::ios::openmode mode) : 
	file(std::fstream(filePath, std::ios::binary | mode))
{
	if (!file.is_open())
		std::cerr << "Failed to open file: " + filePath << '\n';
}

FileManager::~FileManager()
{
	file.close();
}

void FileManager::Write(const char* data, std::streamsize size)
{
	if (!file) 
	{
		std::cerr << "Failed to write to file \n";
		return;
	}

	file.write(data, size);
}

void FileManager::Read(char* data, std::streamsize size)
{
	if (!file) 
	{
		std::cerr << "Failed to read from file \n";
		return;
	}

	file.read(data, size);
}

