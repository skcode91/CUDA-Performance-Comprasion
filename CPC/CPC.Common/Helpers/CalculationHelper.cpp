#include <iostream>
#include <algorithm>

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace CalculationHelper
			{
				double median(double arr[], int n) {
					std::sort(arr, arr + n);
					if (n % 2 == 0) {
						return (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
					}
					else {
						return arr[n / 2];
					}
				}


				double** calculateOnPaddedMatrix(double** matrix, int rows, int cols)
				{
					
					int padSize = 1;
					int paddedRows = rows;
					int paddedCols = cols;
					double *windowArr = new double[8];

					double** resultMatrix = new double* [rows - 2];
					for (int i = 0; i < rows - 2; i++)
					{
						resultMatrix[i] = new double[cols - 2];
					}

					for (int i = padSize; i < paddedRows - padSize; i++)
					{
						for (int j = padSize; j < paddedCols - padSize; j++)
						{
							double sum = 0.0;
							int numNeighbors = 8;
							// 
							if (i == padSize || i == paddedRows - padSize - 1)
							{
								numNeighbors = 5;
							}
							if (j == padSize || j == paddedCols - padSize - 1)
							{
								numNeighbors = 5;
							}
							if ((i == padSize && j == padSize) || (i == padSize && j == paddedCols - padSize - 1) ||
								(i == paddedRows - padSize - 1 && j == padSize) || (i == paddedRows - padSize - 1 && j == paddedCols - padSize - 1))
							{
								numNeighbors = 3;
							}

							windowArr[0] = matrix[i - 1][j - 1];
							windowArr[1] = matrix[i - 1][j];
							windowArr[2] = matrix[i - 1][j + 1];
							windowArr[3] = matrix[i][j - 1];
							windowArr[4] = matrix[i][j + 1];
							windowArr[5] = matrix[i + 1][j - 1];
							windowArr[6] = matrix[i + 1][j];
							windowArr[7] = matrix[i + 1][j + 1];
							resultMatrix[i - padSize][j - padSize] = median(windowArr, 8);
						}
					}

					return resultMatrix;
				}

				double** calculateOnMatrix(double** matrix, int rows, int cols)
				{
					double* windowArr = new double[8];

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
							windowArr[0] = matrix[i - 1][j - 1];
							windowArr[1] = matrix[i - 1][j];
							windowArr[2] = matrix[i - 1][j + 1];
							windowArr[3] = matrix[i][j - 1];
							windowArr[4] = matrix[i][j + 1];
							windowArr[5] = matrix[i + 1][j - 1];
							windowArr[6] = matrix[i + 1][j];
							windowArr[7] = matrix[i + 1][j + 1];
							resultMatrix[i][j] = median(windowArr, 8);
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
