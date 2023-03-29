#pragma once

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace CalculationHelper
			{
				double** calculateOnPaddedMatrix(double** matrix, int rows, int cols);
				double** calculateOnMatrix(double** matrix, int rows, int cols);
				double median(double arr[], int n);
			}
		}
	}
}