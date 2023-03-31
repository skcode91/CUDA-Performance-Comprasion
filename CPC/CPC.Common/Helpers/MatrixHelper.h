#pragma once

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace MatrixHelper
			{
				void fillMatrix(double** matrix, int rows, int cols);

				double*** divideMatrixToZeroPadded(double** matrix, int sizeX, int sizeY, int sizeXDivided, int sizeYDivided, int subMatrixesCount, int overlap, int lastOverlap);
				double*** divideMatrixToSubmatrices(double** matrix, int sizeX, int sizeY, int sizeXDivided, int sizeYDivided, int subMatrixesCount, int lastOverlap);
				double** mergePaddedMatrices(double*** PResultsToMerge, int sizeX, int sizeY, int subMatrixesCount, int sizeXDivided, int overlap, int lastOverlap);
				double** createZeroPaddedMatrix(double** matrix, int rows, int cols);
				double** generateMatrix(int rows, int cols);
				void divideWithOverlap(int arrSize, int subArraysCount, int overlap, int* PElementsInArray, int* POverlapLast);
				void divideWithoutOverlap(int arrSize, int subArraysCount, int* PElementsInArray, int* POverlapLast);

				void deleteArray(double* arr);
				void deleteArray(double** arr, int x);
				void deleteArray(double*** arr, int x, int y);
			}
		}
	}
}