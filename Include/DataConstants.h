#pragma once
#include <string>

class DataConstants
{
public:
	const static std::string trainingImagesPath;
	const static std::string trainingLabelsPath;
	const static std::string testImagesPath;
	const static std::string testLabelsPath;

	const static int NUM_TRAINING_IMAGES;
	const static int NUM_TESTING_IMAGES;
	const static int NUM_OF_PIXELS_PER_IMAGE;
};

