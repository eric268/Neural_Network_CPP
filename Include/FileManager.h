#pragma once

class FileManager
{
public:
	FileManager() = default;
	~FileManager();
	FileManager(std::string filePath, std::ios::openmode mode);

	void Write(const char* data, std::streamsize size);
	void Read(char* data, std::streamsize size);

private:
	std::fstream file;
};