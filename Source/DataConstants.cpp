#include "../Include/pch.h"
#include "../Include/DataConstants.h"

// Training Image Path
const std::string DataConstants::trainingImagesPath = "MNISTData/train-images.idx3-ubyte";
const std::string DataConstants::trainingLabelsPath = "MNISTData/train-labels.idx1-ubyte";

// Testing Image Path
const std::string DataConstants::testImagesPath = "MNISTData/t10k-images.idx3-ubyte";
const std::string DataConstants::testLabelsPath = "MNISTData/t10k-labels.idx1-ubyte";

const int DataConstants::NUM_TRAINING_IMAGES = 60'000;
const int DataConstants::NUM_TESTING_IMAGES = 10'000;
const int DataConstants::NUM_OF_PIXELS_PER_IMAGE = 784;