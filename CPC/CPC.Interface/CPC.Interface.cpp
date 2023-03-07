// CPC.Interface.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <ctime>

#include "../CPC.Common/Helpers/MatrixHelper.cpp"
#include "../CPC.Common/Helpers/CalculationHelper.h"
#include "../CPC.Common/Helpers/BinaryFileHelper.cpp"

int main()
{
	std::cout << "-------------------------------" << std::endl;
	const int rows = 10000;
	const int cols = 10000;
	const std::string matrixFilePath = "./matrix.bin";
	const std::string probeFilePath = "./probe.bin";

	std::cout << "Ilosc pamieci wykorzystywanej przez macierz to " << rows * cols / 1024 / 1024 * sizeof(double) << " Mb" << std::endl;

	std::cout << "Generowanie macierzy..." << std::endl;
	double** matrix = CPC::Common::Helpers::MatrixHelper::generateMatrix(rows, cols);

	//std::cout << "Zapisywanie do pliku..." << std::endl;
	//CPC::Common::Helpers::BinaryFileHelper::saveMatrixToFile(matrix, rows, cols, matrixFilePath);

	//std::cout << "Otwieranie pliku..." << std::endl;
	//double** matrix = CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(matrixFilePath, rows, cols);

	clock_t start = clock();

	std::cout << "Obliczenia..." << std::endl;
	double** results = CPC::Common::Helpers::CalculationHelper::calculateOnMatrix(matrix, rows, cols);

	clock_t end = clock();
	double duration = double(end - start) / CLOCKS_PER_SEC * 1000;

	std::cout << "Obliczenia zakonczono w czasie " << duration << " ms" << std::endl;

	CPC::Common::Helpers::BinaryFileHelper::saveMatrixProbe(probeFilePath, results, rows, cols);
	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, results, rows, cols);
	CPC::Common::Helpers::MatrixHelper::deleteArray(results, rows);

	const int subMatrixesCount = 1000;
	int paddedSizeX, paddedSizeY;

	double*** subMatrixes = CPC::Common::Helpers::MatrixHelper::divideMatrixToZeroPadded(matrix, rows, cols, subMatrixesCount, paddedSizeX, paddedSizeY);;
	double*** resultsToMerge = new double** [subMatrixesCount];

	CPC::Common::Helpers::MatrixHelper::deleteArray(matrix, rows);

	clock_t start2 = clock();

	for (int i = 0; i < subMatrixesCount; i++)
	{
		resultsToMerge[i] = CPC::Common::Helpers::CalculationHelper::calculateOnMatrix(subMatrixes[i], rows, cols);
	}

	double** results2 = CPC::Common::Helpers::MatrixHelper::mergeMatrices(resultsToMerge, subMatrixesCount, paddedSizeX, paddedSizeY, cols);
	clock_t end2 = clock();


	double duration2 = double(end2 - start2) / CLOCKS_PER_SEC * 1000;
	std::cout << "Obliczenia2 zakonczono w czasie " << duration2 << " ms" << std::endl;
	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, results2, rows, cols);

	CPC::Common::Helpers::MatrixHelper::deleteArray(subMatrixes, subMatrixesCount, rows);
	CPC::Common::Helpers::MatrixHelper::deleteArray(resultsToMerge, subMatrixesCount, rows);


	CPC::Common::Helpers::MatrixHelper::deleteArray(matrix, rows);

	std::cout << "-------------------------------" << std::endl;
	return 0;
}

