// CPC.Interface.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <ctime>

#include "../CPC.Common/Helpers/MatrixHelper.cpp"
#include "../CPC.Common/Helpers/BinaryFileHelper.cpp"
#include "../CPC.Common/SingleCore/SingleCore.h"



int main()
{
	// variables
	const int sizeX = 1000;	// cols
	const int sizeY = 1000;	// rows
	const int cycles = 10;

	bool generateMatrix = false;
	bool saveGeneratedMatrixToFile = false;
	bool openMatrixFile = true;

	bool saveProbeToFile = false;

	const std::string matrixFilePath = "D:/matrix.bin";
	const std::string probeFilePath = "D:/probe.txt";

	// allocate memory
	double* PInput = new double[sizeX * sizeY];
	double** input = new double* [sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		input[i] = PInput + i * sizeY;
	}

	double* POutput = new double[sizeX * sizeY];
	double** output = new double* [sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		output[i] = POutput + i * sizeY;
	}
	std::cout << "Allocated memory: 2 x " << sizeY * sizeX / 1024 / 1024 * sizeof(double) << " Mb" << std::endl;


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

	//computing
	clock_t start = clock();

	for (int i = 0; i < cycles; i++)
	{
		std::cout << "computing iteration " << i + 1 << "/" << cycles << std::endl;

		if (i % 2 == 0)
		{
			CPC::Common::SingleCore::SingleCore::medianFilter(input, output, sizeY, sizeX);
		}
		else 
		{
			CPC::Common::SingleCore::SingleCore::medianFilter(output, input, sizeY, sizeX);
		}
	}

	clock_t end = clock();

	// finalize
	double duration = double(end - start) / CLOCKS_PER_SEC * 500;
	std::cout << "Computing finished in " << duration << " ms" << std::endl;

	if (saveProbeToFile)
	{
		CPC::Common::Helpers::BinaryFileHelper::saveMatrixProbe(probeFilePath, cycles % 2 == 0 ? output : input, sizeY, sizeX);
	}

	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, cycles % 2 == 0 ? output : input, sizeY, sizeX);

	//CPC::Common::Helpers::MatrixHelper::deleteArray(input, sizeY);
	//CPC::Common::Helpers::MatrixHelper::deleteArray(output, sizeY);
	return 0;
}

