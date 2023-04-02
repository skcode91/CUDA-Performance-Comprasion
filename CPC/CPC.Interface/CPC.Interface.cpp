// CPC.Interface.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <ctime>
#include <functional>
#include <omp.h>

#include "../CPC.Common/Helpers/MatrixHelper.h"
#include "../CPC.Common/Helpers/BinaryFileHelper.h"
#include "../CPC.Common/SingleCore/SingleCore.h"
#include "../CPC.Common/MultiCore/MultiCore.h"

int main()
{
	// variables
	const int sizeX = 19000; // cols
	const int sizeY = 19000; // rows
	const int cycles = 5;

	bool generateMatrix = true;
	bool saveGeneratedMatrixToFile = true;
	bool openMatrixFile = false;
	bool parallel = true;

	bool saveProbeToFile = true;

	const std::string matrixFilePath = "./matrix.bin";
	const std::string probeFilePath = "./probe.txt";

	// allocate memory
	double *PInput = new double[sizeX * sizeY];
	double **input = new double *[sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		input[i] = PInput + i * sizeY;
	}

	double *POutput = new double[sizeX * sizeY];
	double **output = new double *[sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		output[i] = POutput + i * sizeY;
	}
	std::cout << "allocated memory: 2 x " << sizeY * sizeX / 1024 / 1024 * sizeof(double) << " Mb" << std::endl;

	// creating data
	if (generateMatrix)
	{
		std::cout << "generating data..." << std::endl;
		CPC::Common::Helpers::MatrixHelper::fillMatrix(input, sizeY, sizeX);
	}

	if (saveGeneratedMatrixToFile)
	{
		std::cout << "saving to file..." << std::endl;
		CPC::Common::Helpers::BinaryFileHelper::saveMatrixToFile(input, sizeY, sizeX, matrixFilePath);
	}

	if (openMatrixFile)
	{
		std::cout << "opening file..." << std::endl;
		CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(input, matrixFilePath, sizeY, sizeX);
	}

	// select function to compute
	void (*selectedFunction)(double **, double **, int, int);
	if (parallel)
	{
		selectedFunction = CPC::Common::MultiCore::MultiCore::medianFilter;
	}
	else
	{
		selectedFunction = CPC::Common::SingleCore::SingleCore::medianFilter;
	}

	// computing
	double start = omp_get_wtime();

	for (int i = 0; i < cycles; i++)
	{
		std::cout << "computing iteration " << i + 1 << "/" << cycles << std::endl;

		if (i % 2 == 0)
		{
			selectedFunction(input, output, sizeY, sizeX);
		}
		else
		{
			selectedFunction(output, input, sizeY, sizeX);
		}
	}

	double end = omp_get_wtime();

	// finalize
	double duration = end - start;
	std::cout << "Computing finished in " << duration << " s" << std::endl;

	if (saveProbeToFile)
	{
		CPC::Common::Helpers::BinaryFileHelper::saveMatrixProbe(probeFilePath, cycles % 2 == 0 ? input : output, sizeY, sizeX);
	}

	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, cycles % 2 == 0 ? input : output, sizeY, sizeX);

	// CPC::Common::Helpers::MatrixHelper::deleteArray(input, sizeY);
	// CPC::Common::Helpers::MatrixHelper::deleteArray(output, sizeY);
	return 0;
}
