
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include "../CPC.Common/Helpers/MatrixHelper.cpp"
#include "../CPC.Common/Helpers/CalculationHelper.h"
#include "../CPC.Common/Helpers/BinaryFileHelper.cpp"

__global__ void testMatrix(double* matrix, const int sizeX, const int sizeY)
{
	int i = blockIdx.x;
	int j = threadIdx.x;

	double b = matrix[i * sizeX + j];

	printf("\n%d %d %f", i, j, b);
}

int main()
{
	// firstTest();

		//	//	//	settings	//	//	//	//	//	//

	const int sizeX = 30;	// cols
	const int sizeY = 30;	// rows
	const int subMatrixesCount = 5;

	const std::string matrixFilePath = "D:/matrix.bin";
	const std::string probeFilePath = "D:/probe.txt";

	//	//	//	variables	//	//	//	//	//	//

	int overlap = 2;
	int sizeXDivided, lastOverlap;

	CPC::Common::Helpers::MatrixHelper::divideWithOverlap(sizeX, subMatrixesCount, overlap, &sizeXDivided, &lastOverlap);

	int paddedSizeX = sizeXDivided + 2;
	int paddedSizeY = sizeY + 2;

	//	//	//	//	//	//	//	//	//	//	//	//

	std::cout << "Ilosc pamieci wykorzystywanej przez macierz to " << sizeY * sizeX / 1024 / 1024 * sizeof(double) << " Mb" << std::endl;

	// allocate memory

	double* allocatedMemoryForMatrix = new double[sizeX * sizeY];
	double** matrix = new double* [sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		matrix[i] = allocatedMemoryForMatrix + i * sizeY;
	}


	std::cout << "Otwieranie pliku..." << std::endl;
	CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(matrix, matrixFilePath, sizeY, sizeX);

	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++)
		{
			std::cout << allocatedMemoryForMatrix[i * sizeX + j] << " ";
		}
		std::cout << std::endl;
	}

	double*** subMatrixes = CPC::Common::Helpers::MatrixHelper::divideMatrixToZeroPadded(matrix, sizeX, sizeY, sizeXDivided, sizeY, subMatrixesCount, overlap, lastOverlap);


	// allocate memory on GPU

	double* gpuOutputMatrix;
	double* gpuInputMatrix;

	cudaMalloc(&gpuInputMatrix, subMatrixesCount * paddedSizeX * paddedSizeY * sizeof(double));
	cudaMalloc(&gpuOutputMatrix, sizeX * sizeY * sizeof(double));

	// copy from host to gpu
	cudaMemcpy(gpuInputMatrix, allocatedMemoryForMatrix, subMatrixesCount * paddedSizeX * paddedSizeY * sizeof(double), cudaMemcpyHostToDevice);

	testMatrix << <30, 30 >> > (gpuMatrix, sizeX, sizeY);
	cudaFree(gpuInputMatrix);
	//for (int i = 0; i < sizeX; i++)
	//{
	//	delete[] matrix[i];
	//}
	//delete[] matrix;

	return 0;
}


//
//	int g = 5;
//
//	double*** subMatrixes = CPC::Common::Helpers::MatrixHelper::divideMatrixToZeroPadded(matrix, sizeX, sizeY, sizeXDivided, sizeY, subMatrixesCount, overlap, lastOverlap);
//
//	int size = subMatrixesCount * sizeY * sizeXDivided;
//	double* allocatedMemoryForResultToMerge = new double[size];
//	double*** resultsToMerge = new double** [subMatrixesCount];
//	for (int i = 0; i < subMatrixesCount; i++)
//	{
//		resultsToMerge[i] = new double* [sizeXDivided];
//		for (int j = 0; j < sizeXDivided; j++)
//		{
//			resultsToMerge[i][j] = allocatedMemoryForResultToMerge + i * sizeY * sizeXDivided + j * sizeXDivided;
//		}
//	}
//
//	int dSubMatrixesBytesCount = subMatrixesCount * paddedSizeX * paddedSizeY * sizeof(double);
//	int dResultsToMergeBytesCount = subMatrixesCount * sizeY * sizeXDivided * sizeof(double);
//
//	double* dSubMatrixes;
//	double* dResultsToMerge;
//	cudaMalloc((void**)&dSubMatrixes, dSubMatrixesBytesCount);
//	cudaMalloc((void**)&dResultsToMerge, dResultsToMergeBytesCount);
//
//	cudaMemcpy(dSubMatrixes, subMatrixes, dSubMatrixesBytesCount, cudaMemcpyHostToDevice);
//
//	dim3 threadsPerBlock(16, 16);
//	dim3 numBlocks(paddedSizeX / threadsPerBlock.x + 1, paddedSizeY / threadsPerBlock.y + 1);
//
//	clock_t start2 = clock();
//
//	//calculateOnPaddedMatrix << <numBlocks, threadsPerBlock >> > ( // );
//
//	cudaMemcpy(allocatedMemoryForResultToMerge, dResultsToMerge, dResultsToMergeBytesCount, cudaMemcpyDeviceToHost);
//
//	double** results2 = CPC::Common::Helpers::MatrixHelper::mergeMatrices(resultsToMerge, sizeX, sizeY, subMatrixesCount, sizeXDivided, overlap, lastOverlap);
//	clock_t end2 = clock();
//
//	double duration2 = double(end2 - start2) / CLOCKS_PER_SEC * 1000;
//	std::cout << "Obliczenia zakonczono w czasie " << duration2 << " ms" << std::endl;
//
//	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, results2, sizeY, sizeX);
//
//	cudaFree(dResultsToMerge);
//	cudaFree(dSubMatrixes);
//
//	return 0;
//}
