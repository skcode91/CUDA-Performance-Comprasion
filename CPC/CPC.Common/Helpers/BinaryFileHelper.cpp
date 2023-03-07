#include <iostream>
#include "BinaryFileHelper.h"
#include <fstream>

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace BinaryFileHelper
			{
				void saveMatrixToFile(double** matrix, int rows, int cols, const std::string& filePath) {
					std::ofstream file(filePath, std::ios::binary);

					if (!file.is_open()) {
						std::cerr << "Could not open file " << filePath << " for writing" << std::endl;
						return;
					}

					file.write(reinterpret_cast<char*>(&rows), sizeof(int));
					file.write(reinterpret_cast<char*>(&cols), sizeof(int));
					for (int i = 0; i < rows; ++i) {
						file.write(reinterpret_cast<char*>(matrix[i]), cols * sizeof(double));
					}

					file.close();
				}

				double** readMatrixFromFile(const std::string& filePath, int rows, int cols)
				{
					std::ifstream file(filePath, std::ios::binary);

					if (!file.is_open()) {
						std::cerr << "Could not open file " << filePath << " for reading" << std::endl;
						return nullptr;
					}

					file.read(reinterpret_cast<char*>(&rows), sizeof(int));
					file.read(reinterpret_cast<char*>(&cols), sizeof(int));
					double** matrix = new double* [rows];

					for (int i = 0; i < rows; ++i) {
						matrix[i] = new double[cols];
						file.read(reinterpret_cast<char*>(matrix[i]), cols * sizeof(double));
					}
					file.close();

					return matrix;
				}

				void saveMatrixProbe(const std::string& filePath, double** matrix, int rows, int cols)
				{
					if (rows < 10 || cols < 10)
					{
						std::cout << "Minimalny rozmiar macierzy to 10x10" << std::endl;
					}
					std::cout << "Zapis pliku kontrolnego " << filePath << " ..." << std::endl;
					std::ofstream outfile(filePath);

					outfile << matrix[0][0] << std::endl
						<< matrix[10][10] << std::endl
						<< matrix[rows - 1][0] << std::endl
						<< matrix[rows - 2][0] << std::endl
						<< matrix[rows - 10][1] << std::endl
						<< matrix[rows - 1][0] << std::endl
						<< matrix[10][cols - 10] << std::endl
						<< matrix[rows - 10][cols - 2] << std::endl
						<< matrix[rows - 1][cols - 1];
					outfile.close();
				}

				bool validateMatrixProbe(const std::string& filePath, double** matrix, int rows, int cols)
				{
					std::ifstream infile(filePath);
					if (!infile.is_open())
					{
						std::cout << "Nie udalo sie otworzyc pliku kontrolnego" << std::endl;
						return false;
					}
					const double epsilon = 0.00001;

					double probe[9];

					double matrix_probe[9] = {
						matrix[0][0],
						matrix[10][10],
						matrix[rows - 1][0],
						matrix[rows - 2][0],
						matrix[rows - 10][1],
						matrix[rows - 1][0],
						matrix[10][cols - 10],
						matrix[rows - 10][cols - 2],
						matrix[rows - 1][cols - 1],
					};

					for (int i = 0; i < 9; i++)
					{
						infile >> probe[i];

						if (std::abs(probe[i] - matrix_probe[i]) > epsilon)
						{
							std::cout << "Walidacja zakoñczona niepowodzeniem - macierz wynikowa nieprawid³owa !!!" << std::endl;

							return false;
						}
					}

					infile.close();

					std::cout << "Walidacja zakoñczona powodzeniem" << std::endl;
					return true;
				}
			}
		}
	}
}