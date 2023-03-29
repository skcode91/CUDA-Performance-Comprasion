
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
	int stride;


	printf("\n%d %d %f", i, j);
}

__global__ void calculateOnPaddedMatrix(double* input, double* output, const int paddedSizeX, const int paddedSizeY, const int subMatrixesCount)
{
	int dimersion = blockDim.x;
	int block = blockIdx.x;
	int thread = threadIdx.x;

	int inputOffset = (block * dimersion + thread) * paddedSizeX * paddedSizeY;
	int outputOffset = (block * dimersion + thread) * (paddedSizeX - 2) * (paddedSizeY - 2);
	int padSize = 1;
	//int i = blockIdx.x * threadIdx.x;

	for (int i = padSize; i < paddedSizeX - padSize; i++)
	{
		for (int j = padSize; j < paddedSizeY - padSize; j++)
		{
			int numNeighbors = 8;
			double val = *(input + inputOffset + i * paddedSizeY + j);
			double sum = 0.0;

			if (i == padSize || i == paddedSizeY - padSize - 1)
			{
				numNeighbors = 5;
			}
			if (j == padSize || j == paddedSizeX - padSize - 1)
			{
				numNeighbors = 5;
			}
			if ((i == padSize && j == padSize) || (i == padSize && j == paddedSizeX - padSize - 1) ||
				(i == paddedSizeY - padSize - 1 && j == padSize) || (i == paddedSizeY - padSize - 1 && j == paddedSizeX - padSize - 1))
			{
				numNeighbors = 3;
			}
			// y-1
			sum += *(input + inputOffset + (i - 1) * paddedSizeY + j - 1);
			sum += *(input + inputOffset + (i - 1) * paddedSizeY + j);
			sum += *(input + inputOffset + (i - 1) * paddedSizeY + j + 1);
			// y
			sum += *(input + inputOffset + i * paddedSizeY + j - 1);
			sum += *(input + inputOffset + i * paddedSizeY + j);
			sum += *(input + inputOffset + i * paddedSizeY + j + 1);
			// y + 1
			sum += *(input + inputOffset + (i + 1) * paddedSizeY + j - 1);
			sum += *(input + inputOffset + (i + 1) * paddedSizeY + j);
			sum += *(input + inputOffset + (i + 1) * paddedSizeY + j + 1);



			*(output + outputOffset + (i - padSize) * (paddedSizeY - 2 * padSize) + j - 1 - paddedSizeY - 2 * padSize) = sum / numNeighbors;
		}
	}



	//if (i == 2)
		//printf("\n %d %d %d %d", dimersion, block, thread, i);

}

__global__ void printKernel(double* gpuOutput) {
	for (int i = 0; i < 100; i++) {
		printf("gpuOutput[%d]: %f\n", i, gpuOutput[i]);
	}
}

int main()
{
	// firstTest();

		//	//	//	settings	//	//	//	//	//	//

	const int sizeX = 5000;	// cols
	const int sizeY = 5000;	// rows
	const int subMatrixesCount = 1000;

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

	double* allocatedMemoryForMatrix2 = new double[sizeX * sizeY];
	double** matrix2 = new double* [sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		matrix2[i] = allocatedMemoryForMatrix2 + i * sizeY;
	}


	std::cout << "Otwieranie pliku..." << std::endl;
	CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(matrix, matrixFilePath, sizeY, sizeX);

	//for (int i = 0; i < sizeX; i++)
	//{
	//	for (int j = 0; j < sizeY; j++)
	//	{
	//		std::cout << allocatedMemoryForMatrix[i * sizeX + j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	double*** subMatrixes = CPC::Common::Helpers::MatrixHelper::divideMatrixToZeroPadded(matrix, sizeX, sizeY, sizeXDivided, sizeY, subMatrixesCount, overlap, lastOverlap);


	// allocate memory on GPU

	double* gpuOutputMatrix;
	double* gpuInputMatrix;


	cudaMalloc(&gpuInputMatrix, subMatrixesCount * paddedSizeX * paddedSizeY * sizeof(double));
	cudaMalloc(&gpuOutputMatrix, subMatrixesCount * paddedSizeX * paddedSizeY * sizeof(double));

	// copy from host to gpu
	cudaMemcpy(gpuInputMatrix, subMatrixes[0][0], subMatrixesCount * paddedSizeX * paddedSizeY * sizeof(double), cudaMemcpyHostToDevice);

	int a = 5;
	int threadsPerBlock = 256;
	int blocksPerGrid = (subMatrixesCount + threadsPerBlock - 1) / threadsPerBlock;
	std::cout << blocksPerGrid << std::endl;
	calculateOnPaddedMatrix << <10, 100 >> > (gpuInputMatrix, gpuOutputMatrix, paddedSizeX, paddedSizeY, subMatrixesCount);

	printKernel << <1, 1 >> > (gpuOutputMatrix);
	cudaDeviceSynchronize();


	cudaMemcpy(allocatedMemoryForMatrix2, gpuOutputMatrix, subMatrixesCount * sizeX * sizeY * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 100; i++)

	{
		//std::cout << allocatedMemoryForMatrix2[i] << std::endl;
	}

	int size = subMatrixesCount * sizeY * sizeXDivided;

	//double* allocatedMemoryForResultToMerge = new double[size];
	//double*** resultsToMerge = new double** [subMatrixesCount];
	//for (int i = 0; i < subMatrixesCount; i++)
	//{
	//	resultsToMerge[i] = new double* [sizeY];
	//	for (int j = 0; j < sizeY; j++)
	//	{
	//		resultsToMerge[i][j] = allocatedMemoryForMatrix2 + i * sizeY * sizeX + j * sizeX;
	//	}
	//}



	//double** results2 = CPC::Common::Helpers::MatrixHelper::mergeMatrices(resultsToMerge, sizeX, sizeY, subMatrixesCount, sizeXDivided, overlap, lastOverlap);

	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, matrix2, sizeY, sizeX);




	cudaFree(gpuInputMatrix);
	cudaFree(gpuOutputMatrix);
	//for (int i = 0; i < sizeX; i++)
	//{
	//	delete[] matrix[i];
	//}
	//delete[] matrix;

	return 0;
}

//testMatrix << <30, 30 >> > (gpuMatrix, sizeX, sizeY);

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
