#include <iostream>

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace CalculationHelper
			{
				void calculateOnPaddedMatrix(double**& resultsMatrix, double** matrix, int paddedSizeX, int paddedSizeY)
				{
					int padSize = 1;

					for (int i = padSize; i < paddedSizeX - padSize; i++)
					{
						for (int j = padSize; j < paddedSizeY - padSize; j++)
						{
							double sum = 0.0;
							int numNeighbors = 8;
							if (i == padSize || i == paddedSizeX - padSize - 1)
							{
								numNeighbors = 5;
							}
							if (j == padSize || j == paddedSizeY - padSize - 1)
							{
								numNeighbors = 5;
							}
							if ((i == padSize && j == padSize) || (i == padSize && j == paddedSizeY - padSize - 1) ||
								(i == paddedSizeX - padSize - 1 && j == padSize) || (i == paddedSizeX - padSize - 1 && j == paddedSizeY - padSize - 1))
							{
								numNeighbors = 3;
							}
							sum += matrix[i - 1][j - 1];
							sum += matrix[i - 1][j];
							sum += matrix[i - 1][j + 1];
							sum += matrix[i][j - 1];
							sum += matrix[i][j + 1];
							sum += matrix[i + 1][j - 1];
							sum += matrix[i + 1][j];
							sum += matrix[i + 1][j + 1];
							resultsMatrix[i - padSize][j - padSize] = sum / numNeighbors;
						}
					}

				}

				void calculateOnMatrix(double**& resultsMatrix, double** matrix, int rows, int cols)
				{
					for (int i = 1; i < rows - 1; i++)
					{
						for (int j = 1; j < cols - 1; j++)
						{
							double sum = 0.0;
							sum += matrix[i - 1][j - 1];
							sum += matrix[i - 1][j];
							sum += matrix[i - 1][j + 1];
							sum += matrix[i][j - 1];
							sum += matrix[i][j + 1];
							sum += matrix[i + 1][j - 1];
							sum += matrix[i + 1][j];
							sum += matrix[i + 1][j + 1];
							resultsMatrix[i][j] = sum / 8.0;
						}
					}

					// first and last column
					for (int i = 1; i < rows - 1; i++)
					{
						resultsMatrix[i][0] = (matrix[i - 1][0] + matrix[i - 1][1] + matrix[i][1] + matrix[i + 1][0] + matrix[i + 1][1]) / 5.0;
						resultsMatrix[i][cols - 1] = (matrix[i - 1][cols - 2] + matrix[i - 1][cols - 1] + matrix[i][cols - 2] + matrix[i + 1][cols - 2] + matrix[i + 1][cols - 1]) / 5.0;
					}

					// first and last row
					for (int i = 1; i < cols - 1; i++)
					{
						resultsMatrix[0][i] = (matrix[0][i - 1] + matrix[1][i - 1] + matrix[1][i] + matrix[0][i + 1] + matrix[1][i + 1]) / 5.0;
						resultsMatrix[rows - 1][i] = (matrix[rows - 2][i - 1] + matrix[rows - 1][i - 1] + matrix[rows - 2][i] + matrix[rows - 2][i + 1] + matrix[rows - 1][i + 1]) / 5.0;
					}

					// corners
					resultsMatrix[0][0] = (matrix[0][1] + matrix[1][1] + matrix[1][0]) / 3;
					resultsMatrix[0][cols - 1] = (matrix[0][cols - 2] + matrix[1][cols - 2] + matrix[1][cols - 1]) / 3;
					resultsMatrix[rows - 1][0] = (matrix[rows - 2][0] + matrix[rows - 2][1] + matrix[rows - 1][1]) / 3;
					resultsMatrix[rows - 1][cols - 1] = (matrix[rows - 2][cols - 2] + matrix[rows - 2][cols - 1] + matrix[rows - 1][cols - 2]) / 3;
				}

			}
		}
	}
}
