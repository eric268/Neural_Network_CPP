#pragma once

class FileManager
{
public:
	FileManager() = default;
	FileManager(std::string filePath, std::ios::openmode mode);
	~FileManager();

	void Write(const char* data, std::streamsize size);
	void Read(char* data, std::streamsize size);

private:
	std::fstream file;
};