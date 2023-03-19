// CPC.Interface.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <ctime>

#include "../CPC.Common/Helpers/MatrixHelper.cpp"
#include "../CPC.Common/Helpers/CalculationHelper.h"
#include "../CPC.Common/Helpers/BinaryFileHelper.cpp"



int main()
{
	//	//	//	settings	//	//	//	//	//	//

	const int sizeX = 20000;	// cols
	const int sizeY = 20000;	// rows
	const int subMatrixesCount = 500;

	bool generateMatrix = true;
	bool saveGeneratedMatrixToFile = false;
	bool openMatrixFile = false;

	bool saveProbeToFile = true;

	const std::string matrixFilePath = "D:/matrix.bin";
	const std::string probeFilePath = "D:/probe2.txt";

	//	//	//	variables	//	//	//	//	//	//

	int overlap = 2;
	int sizeXDivided, lastOverlap;

	CPC::Common::Helpers::MatrixHelper::divideWithOverlap(sizeX, subMatrixesCount, overlap, &sizeXDivided, &lastOverlap);

	int paddedSizeX = sizeXDivided + 2;
	int paddedSizeY = sizeY + 2;

	//	//	//	//	//	//	//	//	//	//	//	//

	std::cout << "Ilosc pamieci wykorzystywanej przez macierz to " << sizeY * sizeX / 1024 / 1024 * sizeof(double) << " Mb" << std::endl;

	double* allocatedMemoryForMatrix = new double[sizeX * sizeY];
	double** matrix = new double* [sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		matrix[i] = allocatedMemoryForMatrix + i * sizeY;
	}

	if (generateMatrix)
	{
		std::cout << "Generowanie macierzy..." << std::endl;
		CPC::Common::Helpers::MatrixHelper::fillMatrix(matrix, sizeY, sizeX);
	}

	if (saveGeneratedMatrixToFile)
	{
		std::cout << "Zapisywanie do pliku..." << std::endl;
		CPC::Common::Helpers::BinaryFileHelper::saveMatrixToFile(matrix, sizeY, sizeX, matrixFilePath);
	}

	if (openMatrixFile)
	{
		std::cout << "Otwieranie pliku..." << std::endl;
		CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(matrix, matrixFilePath, sizeY, sizeX);
	}

	std::cout << "Obliczenia bez dzielenia na mniejsze macierze..." << std::endl;

	double* allocatedMemoryForResultsMatrix = new double[sizeX * sizeY];
	double** resultMatrix = new double* [sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		resultMatrix[i] = allocatedMemoryForResultsMatrix + i * sizeY;
	}

	clock_t start = clock();

	CPC::Common::Helpers::CalculationHelper::calculateOnMatrix(resultMatrix, matrix, sizeY, sizeX);

	clock_t end = clock();
	double duration = double(end - start) / CLOCKS_PER_SEC * 500;

	std::cout << "Obliczenia zakonczono w czasie " << duration << " ms" << std::endl;

	if (saveProbeToFile)
	{
		CPC::Common::Helpers::BinaryFileHelper::saveMatrixProbe(probeFilePath, resultMatrix, sizeY, sizeX);
	}

	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, resultMatrix, sizeY, sizeX);

	double*** subMatrixes = CPC::Common::Helpers::MatrixHelper::divideMatrixToZeroPadded(matrix, sizeX, sizeY, sizeXDivided, sizeY, subMatrixesCount, overlap, lastOverlap);

	CPC::Common::Helpers::MatrixHelper::deleteArray(matrix, sizeX);
	CPC::Common::Helpers::MatrixHelper::deleteArray(allocatedMemoryForMatrix);

	int size = subMatrixesCount * sizeY * sizeXDivided;
	double* allocatedMemoryForResultToMerge = new double[size];
	double*** resultsToMerge = new double** [subMatrixesCount];
	for (int i = 0; i < subMatrixesCount; i++)
	{
		resultsToMerge[i] = new double* [sizeXDivided];
		for (int j = 0; j < sizeXDivided; j++)
		{
			resultsToMerge[i][j] = allocatedMemoryForResultToMerge + i * sizeY * sizeXDivided + j * sizeXDivided;
		}
	}

	std::cout << "Obliczenia z dzieleniem na mniejsze macierze, przy ilosci " << subMatrixesCount << " submacierzy..." << std::endl;

	clock_t start2 = clock();

	for (int z = 0; z < subMatrixesCount; z++)
	{
		CPC::Common::Helpers::CalculationHelper::calculateOnPaddedMatrix(resultsToMerge[z], subMatrixes[z], paddedSizeX, paddedSizeY);
	}

	double** results2 = CPC::Common::Helpers::MatrixHelper::mergeMatrices(resultsToMerge, sizeX, sizeY, subMatrixesCount, sizeXDivided, overlap, lastOverlap);
	clock_t end2 = clock();

	double duration2 = double(end2 - start2) / CLOCKS_PER_SEC * 1000;
	std::cout << "Obliczenia zakonczono w czasie " << duration2 << " ms" << std::endl;

	return 0;
}

