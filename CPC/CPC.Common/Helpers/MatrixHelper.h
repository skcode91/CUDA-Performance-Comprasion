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
				double*** divideMatrixToZeroPadded(double** matrix, int rows, int cols, int subMatrixesCount, int& paddedSizeX, int& paddedSizeY);
				double** mergeMatrices(double*** subMatrices, int subMatrixesCount, int paddedSizeX, int paddedSizeY, int numSubCols);
				double** createZeroPaddedMatrix(double** matrix, int rows, int cols);
				double** generateMatrix(int rows, int cols);

				void deleteArray(double* arr);
				void deleteArray(double** arr, int x);
				void deleteArray(double*** arr, int x, int y);
			}
		}
	}
}