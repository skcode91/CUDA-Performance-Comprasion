#pragma once

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace CalculationHelper
			{
				void calculateOnPaddedMatrix(double**& resultsMatrix, double** matrix, int paddedSizeX, int paddedSizeY);
				void calculateOnMatrix(double**& resultsMatrix, double** matrix, int rows, int cols);
			}
		}
	}
}