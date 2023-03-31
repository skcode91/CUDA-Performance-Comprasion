#include <iostream>
#include <algorithm>
#include "SingleCore.h"

namespace CPC
{
	namespace Common
	{
		namespace SingleCore
		{
			namespace SingleCore
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


				double** medianFilterOnPaddedMatrix(double** matrix, int rows, int cols)
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

				void medianFilter(double** PInput, double** POutput, int sizeY, int sizeX)
				{
					double* windowArr = new double[9];

					for (int i = 1; i < sizeY - 1; i++)
					{
						for (int j = 1; j < sizeX - 1; j++)
						{
							double sum = 0.0;
							windowArr[0] = PInput[i - 1][j - 1];
							windowArr[1] = PInput[i - 1][j];
							windowArr[2] = PInput[i - 1][j + 1];
							windowArr[3] = PInput[i][j - 1];
							windowArr[4] = PInput[i][j + 1];
							windowArr[5] = PInput[i + 1][j - 1];
							windowArr[6] = PInput[i + 1][j];
							windowArr[7] = PInput[i + 1][j + 1];
							windowArr[8] = PInput[i][j];

							POutput[i][j] = median(windowArr, 9);
						}
					}

					// first and last column
					for (int i = 1; i < sizeY - 1; i++)
					{
						windowArr[0] = PInput[i - 1][0];
						windowArr[1] = PInput[i - 1][1];
						windowArr[2] = PInput[i][1];
						windowArr[3] = PInput[i + 1][0];
						windowArr[4] = PInput[i + 1][1];
						windowArr[5] = PInput[i][0];
						POutput[i][0] = median(windowArr, 6);

						windowArr[0] = PInput[i - 1][sizeX - 2];
						windowArr[1] = PInput[i - 1][sizeX - 1];
						windowArr[2] = PInput[i][sizeX - 2];
						windowArr[3] = PInput[i + 1][sizeX - 2];
						windowArr[4] = PInput[i + 1][sizeX - 1];
						windowArr[5] = PInput[i][sizeX - 1];
						POutput[i][sizeX - 1] = median(windowArr, 6);
					}

					// first and last row
					for (int i = 1; i < sizeX - 1; i++)
					{
						windowArr[0] = PInput[0][i - 1];
						windowArr[1] = PInput[1][i - 1];
						windowArr[2] = PInput[1][i];
						windowArr[3] = PInput[0][i + 1];
						windowArr[4] = PInput[1][i + 1];
						windowArr[5] = PInput[0][i];

						POutput[0][i] = median(windowArr, 6);

						windowArr[0] = PInput[sizeY - 2][i - 1];
						windowArr[1] = PInput[sizeY - 1][i - 1];
						windowArr[2] = PInput[sizeY - 2][i];
						windowArr[3] = PInput[sizeY - 2][i + 1];
						windowArr[4] = PInput[sizeY - 1][i + 1];
						windowArr[5] = PInput[sizeY - 1][i];
						POutput[sizeY - 1][i] = median(windowArr, 6);
					}

					// corners
					windowArr[0] = PInput[0][1];
					windowArr[1] = PInput[1][1];
					windowArr[2] = PInput[1][0];
					windowArr[3] = PInput[0][0];
					POutput[0][0] = median(windowArr, 4);

					windowArr[0] = PInput[0][sizeX - 2];
					windowArr[1] = PInput[1][sizeX - 2];
					windowArr[2] = PInput[1][sizeX - 1];
					windowArr[3] = PInput[0][sizeX -1];
					POutput[0][sizeX - 1] = median(windowArr, 4);

					windowArr[0] = PInput[sizeY - 2][0];
					windowArr[1] = PInput[sizeY - 2][1];
					windowArr[2] = PInput[sizeY - 1][1];
					windowArr[3] = PInput[sizeY - 1][0];
					POutput[sizeY - 1][0] = median(windowArr, 4);

					windowArr[0] = PInput[sizeY - 2][sizeX - 2];
					windowArr[1] = PInput[sizeY - 2][sizeX - 1];
					windowArr[2] = PInput[sizeY - 1][sizeX - 2];
					windowArr[3] = PInput[sizeY - 1][sizeX - 1];
					POutput[sizeY - 1][sizeX - 1] = median(windowArr, 4);

					delete[] windowArr;
				}

			}
		}
	}
}
