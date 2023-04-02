namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace MatrixHelper
			{
				double median(double arr[], int n);
				void fillMatrix(double **matrix, int rows, int cols);

				double ***divideMatrixToZeroPadded(double **matrix, int sizeX, int sizeXDivided, int sizeYDivided, int subMatrixesCount, int overlap);
				double **createZeroPaddedMatrix(double **matrix, int rows, int cols);
				double **generateMatrix(int rows, int cols);
				void divideWithOverlap(int arrSize, int subArraysCount, int overlap, int *PElementsInArray, int *POverlapLast);

				void deleteArray(double *arr);
				void deleteArray(double **arr, int x);
				void deleteArray(double ***arr, int x, int y);
			}
		}
	}
}