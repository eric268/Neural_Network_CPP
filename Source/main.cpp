#include "../Include/pch.h"

void ReadMNIST(std::string, std::vector<std::vector<double>>&);
void DisplayImage(std::vector<std::vector<double>>&, int);

int main()
{
    std::vector<std::vector<double>> imageArray(60'000, std::vector<double>(784));
    ReadMNIST("MNISTData/train-images.idx3-ubyte", imageArray);
    int counter = 0;
    char n;
    while (std::cin >> n)
    {
        if (n == 'b')
            break;
        system("CLS");
        DisplayImage(imageArray, counter++);
    }
    return 0;
}

void DisplayImage(std::vector<std::vector<double>>& arr, int numberIndex)
{
    int counter = 0;
    for (int i = 0; i < 784; i++)
    {
        if (i % 28 == 0)
            std::cout << "\n";
        if (arr[numberIndex][i] > 0)
        {
            std::cout << "*";
        }
        else
        {
            std::cout << " ";
        }
    }
}

int ReverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void ReadMNIST(std::string path, std::vector<std::vector<double>>& arr)
{
    std::ifstream file(path, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
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
                    arr[i][(n_rows * r) + c] = (double)temp;
                }
            }
        }
    }
}