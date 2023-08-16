#include "../Include/pch.h"
#include "../Include/DataManager.h"
#include "../Include/DataConstants.h"

DataManager::DataManager()
{
	trainingData = LoadImageData(DataConstants::trainingImagesPath, DataConstants::trainingLabelsPath, DataConstants::NUM_TRAINING_IMAGES);
	testingData = LoadImageData(DataConstants::testImagesPath, DataConstants::testLabelsPath, DataConstants::NUM_TESTING_IMAGES);
}

DataManager::~DataManager()
{

}

int DataManager::ReverseInt(int i)
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::pair<std::vector<double>, int>> DataManager::LoadImageData(std::string path, std::string labelsPath, int totalImages)
{
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	std::vector<std::vector<double>> imageData(totalImages, std::vector<double>(DataConstants::NUM_OF_PIXELS_PER_IMAGE));
	std::vector<int> labelData(totalImages, -1);

	//Image Reader
	std::ifstream file(path, std::ios::binary);
	if (file.is_open())
	{
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for (int i = 0; i < number_of_images; ++i)
		{
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					imageData[i][(n_rows * r) + c] = (double)temp;
				}
			}
		}
	}
	//Label Reader
	std::ifstream file2(labelsPath, std::ios::binary);
	if (file2.is_open())
	{
		file2.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file2.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		for (int i = 0; i < number_of_images; i++)
		{
			char v;
			file2.read(&v, 1);
			labelData[i] = (int)v;
		}
	}

	std::vector<std::pair<std::vector<double>, int>> result(totalImages);
	for (int i = 0; i < totalImages; i++)
		result[i] = std::make_pair(imageData[i], labelData[i]);

	return result;
}

void DataManager::ShuffleTrainingData()
{
	auto rd = std::random_device{};
	auto rng = std::default_random_engine{ rd() };
	std::shuffle(std::begin(trainingData), std::end(trainingData), rng);
}