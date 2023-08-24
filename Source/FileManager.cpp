#include "../Include/FileManager.h"

void FileManager::Write(const char* data, std::streamsize size) 
{
	if (!file) {
		throw std::runtime_error("File stream is in an invalid state");
	}

	// Save current position
	auto pos = file.tellp();
	file.write(data, size);

	if (file.fail()) {
		// Clear fail state
		file.clear();
		// Roll back to original position
		file.seekp(pos);
		throw std::runtime_error("Write operation failed");
	}
}

void FileManager::Read(char* data, std::streamsize size)
{
	if (!file) {
		throw std::runtime_error("File stream is in an invalid state");
	}

	// Save current position
	auto pos = file.tellg();

	std::vector<char> tempData(size);
	file.read(tempData.data(), size);

	if (file.fail() && !file.eof()) {
		// Clear fail state
		file.clear();
		// Roll back to original position
		file.seekg(pos);
		throw std::runtime_error("Read operation failed");
	}

	std::copy(tempData.begin(), tempData.end(), data);
}

