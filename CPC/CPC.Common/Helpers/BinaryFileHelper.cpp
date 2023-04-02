#include <iostream>
#include "./BinaryFileHelper.h"
#include <fstream>

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace BinaryFileHelper
			{
				void saveMatrixToFile(double **matrix, int rows, int cols, const std::string &filePath)
				{
					std::ofstream file(filePath, std::ios::binary);

					if (!file.is_open())
					{
						std::cerr << "Could not open file " << filePath << " for writing" << std::endl;
						return;
					}

					file.write(reinterpret_cast<char *>(&rows), sizeof(int));
					file.write(reinterpret_cast<char *>(&cols), sizeof(int));
					for (int i = 0; i < rows; ++i)
					{
						file.write(reinterpret_cast<char *>(matrix[i]), cols * sizeof(double));
					}

					file.close();
				}

				void readMatrixFromFile(double **matrix, const std::string &filePath, int rows, int cols)
				{
					std::ifstream file(filePath, std::ios::binary);

					if (!file.is_open())
					{
						std::cerr << "Could not open file " << filePath << " for reading" << std::endl;
					}

					file.read(reinterpret_cast<char *>(&rows), sizeof(int));
					file.read(reinterpret_cast<char *>(&cols), sizeof(int));

					for (int i = 0; i < rows; ++i)
					{
						file.read(reinterpret_cast<char *>(matrix[i]), cols * sizeof(double));
					}
					file.close();
				}

				void saveMatrixProbe(const std::string &filePath, double **matrix, int rows, int cols)
				{
					if (rows < 10 || cols < 10)
					{
						std::cout << "Minimalny rozmiar macierzy to 10x10" << std::endl;
					}
					std::cout << "Zapis pliku kontrolnego " << filePath << " ..." << std::endl;
					std::ofstream outfile(filePath);

					outfile << matrix[0][0] << std::endl
							<< matrix[rows - 1][cols - 1] << std::endl
							<< matrix[rows - 1][0] << std::endl
							<< matrix[0][cols - 1] << std::endl
							<< matrix[1][1] << std::endl
							<< matrix[rows / 2][cols / 2] << std::endl
							<< matrix[rows - 2][cols - 2] << std::endl
							<< matrix[1][0] << std::endl
							<< matrix[rows - 2][0] << std::endl
							<< matrix[1][cols - 1] << std::endl
							<< matrix[rows - 2][cols - 1] << std::endl
							<< matrix[0][1] << std::endl
							<< matrix[0][cols - 2] << std::endl
							<< matrix[rows - 1][1] << std::endl
							<< matrix[rows - 1][cols - 2] << std::endl
							<< matrix[10][10] << std::endl
							<< matrix[rows - 10][cols - 10];
					outfile.close();
				}

				bool validateMatrixProbe(const std::string &filePath, double **matrix, int rows, int cols)
				{
					std::ifstream infile(filePath);
					if (!infile.is_open())
					{
						std::cout << "Nie udalo sie otworzyc pliku kontrolnego" << std::endl;
						return false;
					}
					const double epsilon = 0.00001;

					double probe[17];
					double matrix_probe[17];

					// corner element
					matrix_probe[0] = matrix[0][0];

					// corner element
					matrix_probe[1] = matrix[rows - 1][cols - 1];

					// corner element
					matrix_probe[2] = matrix[rows - 1][0];

					// corner element
					matrix_probe[3] = matrix[0][cols - 1];

					// first element of main kernel
					matrix_probe[4] = matrix[1][1];

					// middle element
					matrix_probe[5] = matrix[rows / 2][cols / 2];

					// last element of main kernel
					matrix_probe[6] = matrix[rows - 2][cols - 2];

					// first element of first column kernel
					matrix_probe[7] = matrix[1][0];

					// last element of first column kernel
					matrix_probe[8] = matrix[rows - 2][0];

					// first element of last column kernel
					matrix_probe[9] = matrix[1][cols - 1];

					// last element of last column kernel
					matrix_probe[10] = matrix[rows - 2][cols - 1];

					// first element of first row kernel
					matrix_probe[11] = matrix[0][1];

					// last element of first row kernel
					matrix_probe[12] = matrix[0][cols - 2];

					// first element of last row kernel
					matrix_probe[13] = matrix[rows - 1][1];

					// last element of last row kernel
					matrix_probe[14] = matrix[rows - 1][cols - 2];

					// random
					matrix_probe[15] = matrix[10][10];

					// random
					matrix_probe[16] = matrix[rows - 10][cols - 10];

					for (int i = 0; i < 17; i++)
					{
						infile >> probe[i];
						if (std::abs(probe[i] - matrix_probe[i]) > epsilon)
						{
							infile.close();
							std::cout << "Walidacja zakonczona niepowodzeniem" << std::endl;
							return false;
						}
					}

					infile.close();
					std::cout << "Walidacja zakonczona powodzeniem" << std::endl;
					return true;
				}
			}
		}
	}
}