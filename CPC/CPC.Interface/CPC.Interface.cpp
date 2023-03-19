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

	const int sizeX = 50;	// cols
	const int sizeY = 50;	// rows
	const int subMatrixesCount = 22;

	const std::string matrixFilePath = "./matrix.bin";
	const std::string probeFilePath = "./probes.bin";

	//	//	//	variables	//	//	//	//	//	//

	int overlap = 2;
	int sizeXDivided, lastOverlap;

	CPC::Common::Helpers::MatrixHelper::divideWithOverlap(sizeX, subMatrixesCount, overlap, &sizeXDivided, &lastOverlap);

	int paddedSizeX = sizeXDivided + 2;
	int paddedSizeY = sizeY + 2;

	//	//	//	//	//	//	//	//	//	//	//	//




	std::cout << "Ilosc pamieci wykorzystywanej przez macierz to " << sizeY * sizeX / 1024 / 1024 * sizeof(double) << " Mb" << std::endl;

	std::cout << "Generowanie macierzy..." << std::endl;
	double** matrix = CPC::Common::Helpers::MatrixHelper::generateMatrix(sizeY, sizeX);

	//std::cout << "Zapisywanie do pliku..." << std::endl;
	//CPC::Common::Helpers::BinaryFileHelper::saveMatrixToFile(matrix, sizeY, sizeX, matrixFilePath);

	//std::cout << "Otwieranie pliku..." << std::endl;
	//double** matrix = CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(matrixFilePath, sizeY, sizeX);

	std::cout << std::endl;

	clock_t start = clock();

	std::cout << "Obliczenia..." << std::endl;
	double** results = CPC::Common::Helpers::CalculationHelper::calculateOnMatrix(matrix, sizeY, sizeX);

	clock_t end = clock();
	double duration = double(end - start) / CLOCKS_PER_SEC * 500;

	//std::cout << "Obliczenia zakonczono w czasie " << duration << " ms" << std::endl;
	std::cout << "WYNIKI " << std::endl << std::endl;
	for (int i = 0; i < (sizeY < 20 ? sizeY : 20); i++)
	{
		for (int j = 0; j < (sizeX < 20 ? sizeX : 20); j++)
		{
			std::cout << results[i][j] << " ";
		}
		std::cout << std::endl;
	}
	CPC::Common::Helpers::BinaryFileHelper::saveMatrixProbe(probeFilePath, results, sizeY, sizeX);
	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, results, sizeY, sizeX);
	//CPC::Common::Helpers::MatrixHelper::deleteArray(results, sizeY);


	double*** subMatrixes = CPC::Common::Helpers::MatrixHelper::divideMatrixToZeroPadded(matrix, sizeX, sizeY, sizeXDivided, sizeY, subMatrixesCount, overlap, lastOverlap);

	double* allocatedMemoryForResultToMerge = new double[subMatrixesCount * sizeY * sizeX];
	double*** resultsToMerge = new double** [subMatrixesCount];
	for (int i = 0; i < subMatrixesCount; i++)
	{
		resultsToMerge[i] = new double* [sizeY];
		for (int j = 0; j < sizeY; j++)
		{
			resultsToMerge[i][j] = allocatedMemoryForResultToMerge + i * sizeY * sizeX + j * sizeX;
		}
	}

	//CPC::Common::Helpers::MatrixHelper::deleteArray(matrix, sizeY);

	clock_t start2 = clock();

	for (int z = 0; z < subMatrixesCount; z++)
	{
		//std::cout << " Do laczenia " << z << std::endl << std::endl;
		resultsToMerge[z] = CPC::Common::Helpers::CalculationHelper::calculateOnPaddedMatrix(subMatrixes[z], paddedSizeX, paddedSizeY);
		//for (int i = 0; i < paddedSizeX - 2; i++)
		//{
		//	for (int j = 0; j < paddedSizeY - 2; j++)
		//	{
		//		std::cout << resultsToMerge[z][i][j] << " ";
		//	}
		//	std::cout << std::endl;
		//}
	}

	std::cout << std::endl << "--------------" << std::endl;



	double** results2 = CPC::Common::Helpers::MatrixHelper::mergeMatrices(resultsToMerge, sizeX, sizeY, subMatrixesCount, sizeXDivided, overlap, lastOverlap);
	clock_t end2 = clock();


	for (int i = 0; i < (sizeY < 20 ? sizeY : 20); i++)
	{
		for (int j = 0; j < (sizeX < 20 ? sizeX : 20); j++)
		{
			std::cout << results2[i][j] << " ";
		}
		std::cout << std::endl;
	}

	double duration2 = double(end2 - start2) / CLOCKS_PER_SEC * 1000;
	std::cout << "Obliczenia2 zakonczono w czasie " << duration2 << " ms" << std::endl;
	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, results2, sizeY, sizeX);

	//CPC::Common::Helpers::MatrixHelper::deleteArray(subMatrixes, subMatrixesCount, sizeY);
	//CPC::Common::Helpers::MatrixHelper::deleteArray(resultsToMerge, subMatrixesCount, sizeY);


	CPC::Common::Helpers::MatrixHelper::deleteArray(matrix, sizeY);

	std::cout << "-------------------------------" << std::endl;
	return 0;
}

