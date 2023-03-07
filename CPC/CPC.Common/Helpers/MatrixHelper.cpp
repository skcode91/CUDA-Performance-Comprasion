#include "MatrixHelper.h"
#include <iostream>

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace MatrixHelper
			{
				void fillMatrix(double** matrix, int rows, int cols)
				{
					srand(time(NULL));
					for (int i = 0; i < rows; i++)
					{
						matrix[i] = new double[cols];
						for (int j = 0; j < cols; j++)
						{
							matrix[i][j] = (double)rand() / RAND_MAX;
						}
					}
				}

				double **generateMatrix(int rows, int cols)
				{
					double** matrix = new double* [rows];
					fillMatrix(matrix, rows, cols);

					return matrix;
				}

				double** createZeroPaddedMatrix(double** matrix, int rows, int cols)
				{
					int paddedRows = rows + 2;
					int paddedCols = cols + 2;
					double** paddedMatrix = new double* [paddedRows];

					for (int i = 0; i < paddedRows; i++)
					{
						paddedMatrix[i] = new double[paddedCols];
					}

					for (int i = 0; i < paddedRows; i++)
					{
						for (int j = 0; j < paddedCols; j++)
						{
							if (i == 0 || i == paddedRows - 1 || j == 0 || j == paddedCols - 1)
							{
								paddedMatrix[i][j] = 0.0;
							}
							else
							{
								paddedMatrix[i][j] = matrix[i - 1][j - 1];
							}
						}
					}

					return paddedMatrix;
				}

				double*** divideMatrixToZeroPadded(double** matrix, int rows, int cols, int subMatrixesCount, int& paddedSizeX, int& paddedSizeY)
				{
					int sizeX = cols % subMatrixesCount == 0 ? cols / subMatrixesCount : cols / subMatrixesCount + 1;

					paddedSizeX = sizeX + 2;
					paddedSizeY = cols + 2;

					double*** subMatrices = new double** [subMatrixesCount];
					for (int i = 0; i < subMatrixesCount; i++)
					{
						subMatrices[i] = new double* [paddedSizeX];
						for (int x = 0; x < paddedSizeX; x++)
						{
							subMatrices[i][x] = new double[paddedSizeY];
						}
					}

					int startAbsoluteX = 0;

					for (int i = 0; i < subMatrixesCount; i++)
					{
						for (int x = 0; x < paddedSizeX; x++)
						{
							for (int y = 0; y < paddedSizeY; y++)
							{
								if (x == 0 || x == paddedSizeX - 1 || y == 0 || y == paddedSizeY - 1)
								{
									subMatrices[i][x][y] = 0.0;
								}
								else
								{
									subMatrices[i][x][y] = matrix[startAbsoluteX + x - 1][y - 1];
								}
							}
						}
						startAbsoluteX = startAbsoluteX + sizeX < cols ? startAbsoluteX + sizeX : cols - sizeX;
					}

					return subMatrices;
				}

				double** mergeMatrices(double*** subMatrices, int subMatrixesCount, int paddedSizeX, int paddedSizeY, int numSubCols)
				{
					int rowSize = paddedSizeX - 2;
					double** mergedMatrix = new double* [rowSize * subMatrixesCount];

					for (int i = 0; i < rowSize * subMatrixesCount; i++)
					{
						mergedMatrix[i] = new double[numSubCols];
					}

					for (int i = 0; i < subMatrixesCount; i++)
					{
						int rowOffset = i * rowSize;
						for (int x = 1; x <= rowSize; x++)
						{
							int mergedRowIndex = rowOffset + x - 1;
							for (int y = 1; y <= numSubCols; y++)
							{
								mergedMatrix[mergedRowIndex][y - 1] = subMatrices[i][x][y];
							}
						}
					}

					return mergedMatrix;
				}

				void deleteArray(double* arr)
				{
					delete[] arr;
				}

				void deleteArray(double** arr, int x)
				{
					for (int i = 0; i < x; i++) {
						delete[] arr[i];
					}

					delete[] arr;
				}

				void deleteArray(double*** arr, int x, int y)
				{
					for (int i = 0; i < x; i++) {
						for (int j = 0; j < y; j++) {
							delete[] arr[i][j];
						}
						delete[] arr[i];
					}

					delete[] arr;
				}
			}
		}
	}
}
