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

				double** generateMatrix(int rows, int cols)
				{
					double** matrix = new double* [rows];
					fillMatrix(matrix, rows, cols);

					return matrix;
				}

				void divideWithOverlap(int arrSize, int subArraysCount, int overlap, int* PElementsInArray, int* POverlapLast)
				{
					int totalOverlap = overlap * (subArraysCount - 1);
					int totalElements = arrSize + totalOverlap;

					*PElementsInArray = totalElements / subArraysCount + 1;

					*POverlapLast = totalElements % subArraysCount;
					if (*POverlapLast == 0)
					{
						*POverlapLast = overlap;
						*PElementsInArray -= *PElementsInArray - overlap < 0 ? 0 : *PElementsInArray - overlap;
					}
					else
					{
						*POverlapLast = *POverlapLast - overlap < 0 ? 0 : *POverlapLast + overlap;
					}
				}

				double** createZeroPaddedMatrix(double** matrix, int sizeX, int sizeY)
				{
					int paddedSizeX = sizeX + 2;
					int paddedSizeY = sizeY + 2;

					double* allocatedMemory = new double[paddedSizeX * paddedSizeY];
					double** paddedMatrix = new double* [paddedSizeX];
					for (int i = 0; i < paddedSizeX; i++)
					{
						paddedMatrix[i] = allocatedMemory + i * paddedSizeY;
					}

					for (int i = 0; i < paddedSizeX; i++)
					{
						for (int j = 0; j < paddedSizeX; j++)
						{
							if (i == 0 || i == paddedSizeX - 1 || j == 0 || j == paddedSizeX - 1)
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

				double*** divideMatrixToZeroPadded(double** matrix, int sizeX, int sizeY, int sizeXDivided, int sizeYDivided, int subMatrixesCount, int overlap, int lastOverlap)
				{
					double* PMatrix = *matrix;
					int paddedSizeX = sizeXDivided + 2;
					int paddedSizeY = sizeYDivided + 2;

					double* allocatedMemory = new double[subMatrixesCount * paddedSizeX * paddedSizeY];
					double*** subMatrices = new double** [subMatrixesCount];

					for (int i = 0; i < subMatrixesCount; i++) {
						subMatrices[i] = new double* [paddedSizeX];
						for (int j = 0; j < paddedSizeX; j++) {
							subMatrices[i][j] = allocatedMemory + i * paddedSizeX * paddedSizeY + j * paddedSizeY;
						}
					}

					int startAbsoluteX = 0;

					for (int i = 0; i < subMatrixesCount; i++)
					{
						int absoluteOffset = i * paddedSizeX * paddedSizeY;

						for (int x = 0; x < paddedSizeX; x++)
						{
							int xOffset = x * paddedSizeY;
							for (int y = 0; y < paddedSizeY; y++)
							{
								int yOffset = y;
								int offset = absoluteOffset + xOffset + yOffset;

								if (x == 0 || x == paddedSizeX - 1 || y == 0 || y == paddedSizeY - 1)
								{
									*(allocatedMemory + offset) = 0.0;
								}
								else
								{
									double w = matrix[startAbsoluteX + x - 1][y - 1];
									*(allocatedMemory + offset) = matrix[startAbsoluteX + x - 1][y - 1];
								}
							}
						}
						startAbsoluteX += (sizeXDivided - overlap);
						if (startAbsoluteX > sizeX - sizeXDivided)
						{
							startAbsoluteX = sizeX - sizeXDivided;
						}
					}

					return subMatrices;
				}

				double** mergeMatrices(double*** PResultsToMerge, int sizeX, int sizeY, int subMatrixesCount, int sizeXDivided, int overlap, int lastOverlap)
				{
					double* allocatedData = new double[sizeX * sizeY];
					double** mergedMatrix = new double* [sizeX];

					for (int i = 0; i < sizeX; i++) {
						mergedMatrix[i] = allocatedData + i * sizeY;
					}

					//without last matrix
					for (int i = 0; i < subMatrixesCount - 1; i++)
					{
						int absoluteOffset = i * (sizeXDivided * sizeY - (overlap * sizeY));

						for (int x = i > 0 ? 1 : 0; x < sizeXDivided; x++)
						{
							int xOffset = x * sizeY;

							for (int y = 0; y < sizeY; y++)
							{
								int yOffset = y;
								int offset = absoluteOffset + xOffset + yOffset;

								*(allocatedData + offset) = PResultsToMerge[i][x][y];
							}
						}
					}

					// last matrix
					int absoluteOffset = (sizeX - sizeXDivided) * sizeY;
					for (int x = 0; x < sizeXDivided; x++)
					{
						int offsetX = x * sizeY;
						for (int y = 0; y < sizeY; y++)
						{
							int offsetY = y;
							int offset = absoluteOffset + offsetX + offsetY;
							*(allocatedData + offset) = PResultsToMerge[subMatrixesCount - 1][x][y];
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
