#pragma once

namespace CPC
{
	namespace Common
	{
		namespace SingleCore
		{
			namespace SingleCore
			{
				double** medianFilterOnPaddedMatrix(double** matrix, int rows, int cols);
				void medianFilter(double** PInput, double ** POutput, int sizeY, int sizeX);
				double median(double arr[], int n);
			}
		}
	}
}