#pragma once

namespace CPC
{
	namespace Common
	{
		namespace Helpers
		{
			namespace BinaryFileHelper
			{
				void saveMatrixToFile(double** matrix, int rows, int cols, const std::string& filePath);
				double** readMatrixFromFile(const std::string& filename, int rows, int cols);

				void saveMatrixProbe(const std::string& filename, double** matrix, int rows, int cols);
				bool validateMatrixProbe(const std::string& filePath, double** matrix, int rows, int cols);
			}
		}
	}
}