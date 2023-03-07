#include <iostream>

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace CalculationHelper
			{
				double** calculateOnPaddedMatrix(double** matrix, int rows, int cols)
				{
					double** resultMatrix = new double* [rows];
					for (int i = 0; i < rows; i++)
					{
						resultMatrix[i] = new double[cols];
					}

					for (int i = 0; i < rows; i++)
					{
						for (int j = 0; j < cols; j++)
						{
							double sum = 0.0;
							sum += matrix[i][j];
							sum += matrix[i][j + 1];
							sum += matrix[i][j + 2];
							sum += matrix[i + 1][j];
							sum += matrix[i + 1][j + 2];
							sum += matrix[i + 2][j];
							sum += matrix[i + 2][j + 1];
							sum += matrix[i + 2][j + 2];

							resultMatrix[i][j] = sum / 8.0;
						}
					}

					return resultMatrix;
				}

				double** calculateOnMatrix(double** matrix, int rows, int cols)
				{
					double** resultMatrix = new double* [rows];
					for (int i = 0; i < rows; i++)
					{
						resultMatrix[i] = new double[cols];
					}

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
							resultMatrix[i][j] = sum / 8.0;
						}
					}

					// first and last column
					for (int i = 1; i < rows - 1; i++)
					{
						resultMatrix[i][0] = (matrix[i - 1][0] + matrix[i - 1][1] + matrix[i][1] + matrix[i + 1][0] + matrix[i + 1][1]) / 5.0;
						resultMatrix[i][cols - 1] = (matrix[i - 1][cols - 2] + matrix[i - 1][cols - 1] + matrix[i][cols - 2] + matrix[i + 1][cols - 2] + matrix[i + 1][cols - 1]) / 5.0;
					}

					// first and last row
					for (int i = 1; i < cols - 1; i++)
					{
						resultMatrix[0][i] = (matrix[0][i - 1] + matrix[1][i - 1] + matrix[1][i] + matrix[0][i + 1] + matrix[1][i + 1]) / 5.0;
						resultMatrix[rows - 1][i] = (matrix[rows - 2][i - 1] + matrix[rows - 1][i - 1] + matrix[rows - 2][i] + matrix[rows - 2][i + 1] + matrix[rows - 1][i + 1]) / 5.0;
					}

					// corners
					resultMatrix[0][0] = (matrix[0][1] + matrix[1][1] + matrix[1][0]) / 3;
					resultMatrix[0][cols - 1] = (matrix[0][cols - 2] + matrix[1][cols - 2] + matrix[1][cols - 1]) / 3;
					resultMatrix[rows - 1][0] = (matrix[rows - 2][0] + matrix[rows - 2][1] + matrix[rows - 1][1]) / 3;
					resultMatrix[rows - 1][cols - 1] = (matrix[rows - 2][cols - 2] + matrix[rows - 2][cols - 1] + matrix[rows - 1][cols - 2]) / 3;

					return resultMatrix;
				}

			}
		}
	}
}