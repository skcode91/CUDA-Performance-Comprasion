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
					double *windowArr = new double[9];

					double** resultMatrix = new double* [rows - 2];
					for (int i = 0; i < rows - 2; i++)
					{
						resultMatrix[i] = new double[cols - 2];
					}

					for (int i = padSize; i < paddedRows - padSize; i++)
					{
						for (int j = padSize; j < paddedCols - padSize; j++)
						{
							windowArr[0] = matrix[i - 1][j - 1];
							windowArr[1] = matrix[i - 1][j];
							windowArr[2] = matrix[i - 1][j + 1];
							windowArr[3] = matrix[i][j - 1];
							windowArr[4] = matrix[i][j + 1];
							windowArr[5] = matrix[i + 1][j - 1];
							windowArr[6] = matrix[i + 1][j];
							windowArr[7] = matrix[i + 1][j + 1];
							windowArr[8] = matrix[i][j];
							resultMatrix[i - padSize][j - padSize] = median(windowArr, 9);
						}
					}

					return resultMatrix;
				}

				double** calculateOnMatrix(double** matrix, int rows, int cols)
				{
					double* windowArr = new double[9];

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
							windowArr[8] = matrix[i][j];

							resultMatrix[i][j] = median(windowArr, 9);
						}
					}

					// first and last column
					for (int i = 1; i < rows - 1; i++)
					{
						windowArr[0] = matrix[i - 1][0];
						windowArr[1] = matrix[i - 1][1];
						windowArr[2] = matrix[i][1];
						windowArr[3] = matrix[i + 1][0];
						windowArr[4] = matrix[i + 1][1];
						windowArr[5] = matrix[i][0];
						resultMatrix[i][0] = median(windowArr, 6);

						windowArr[0] = matrix[i - 1][cols - 2];
						windowArr[1] = matrix[i - 1][cols - 1];
						windowArr[2] = matrix[i][cols - 2];
						windowArr[3] = matrix[i + 1][cols - 2];
						windowArr[4] = matrix[i + 1][cols - 1];
						windowArr[5] = matrix[i][cols - 1];
						resultMatrix[i][cols - 1] = median(windowArr, 6);
					}

					// first and last row
					for (int i = 1; i < cols - 1; i++)
					{
						windowArr[0] = matrix[0][i - 1];
						windowArr[1] = matrix[1][i - 1];
						windowArr[2] = matrix[1][i];
						windowArr[3] = matrix[0][i + 1];
						windowArr[4] = matrix[1][i + 1];
						windowArr[5] = matrix[0][i];

						resultMatrix[0][i] = median(windowArr, 6);

						windowArr[0] = matrix[rows - 2][i - 1];
						windowArr[1] = matrix[rows - 1][i - 1];
						windowArr[2] = matrix[rows - 2][i];
						windowArr[3] = matrix[rows - 2][i + 1];
						windowArr[4] = matrix[rows - 1][i + 1];
						windowArr[5] = matrix[rows - 1][i];
						resultMatrix[rows - 1][i] = median(windowArr, 6);
					}

					// corners
					windowArr[0] = matrix[0][1];
					windowArr[1] = matrix[1][1];
					windowArr[2] = matrix[1][0];
					windowArr[3] = matrix[0][0];
					resultMatrix[0][0] = median(windowArr, 4);

					windowArr[0] = matrix[0][cols - 2];
					windowArr[1] = matrix[1][cols - 2];
					windowArr[2] = matrix[1][cols - 1];
					windowArr[3] = matrix[0][cols-1];
					resultMatrix[0][cols - 1] = median(windowArr, 4);

					windowArr[0] = matrix[rows - 2][0];
					windowArr[1] = matrix[rows - 2][1];
					windowArr[2] = matrix[rows - 1][1];
					windowArr[3] = matrix[rows - 1][0];
					resultMatrix[rows - 1][0] = median(windowArr, 4);

					windowArr[0] = matrix[rows - 2][cols - 2];
					windowArr[1] = matrix[rows - 2][cols - 1];
					windowArr[2] = matrix[rows - 1][cols - 2];
					windowArr[3] = matrix[rows - 1][cols - 1];
					resultMatrix[rows - 1][cols - 1] = median(windowArr, 4);

					return resultMatrix;
				}

			}
		}
	}
}
