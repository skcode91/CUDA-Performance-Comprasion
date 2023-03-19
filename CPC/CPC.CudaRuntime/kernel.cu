
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include "../CPC.Common/Helpers/MatrixHelper.cpp"
#include "../CPC.Common/Helpers/CalculationHelper.h"
#include "../CPC.Common/Helpers/BinaryFileHelper.cpp"


cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void helloFromGpu()
{
	printf("Hello World! frin thread [%d, %d] \
        From device\n", threadIdx.x, blockIdx.x);
}

__global__ void square(float* d_out, float* d_in)
{
	int idx = threadIdx.x;
	float x = d_in[idx];
	d_out[idx] = x * x;
}

__global__ void calculateOnPaddedMatrix(double* dArrOut, double* dArrIn, int paddedSizeX, int paddedSizeY)
{
		int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
		int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

		if (i >= 1 && i < paddedSizeX - 1 && j >= 1 && j < paddedSizeY - 1)
		{
			//double sum = 0.0;
			//int numNeighbors = 8;
			//if (i == 1 || i == paddedSizeX - 2 || j == 1 || j == paddedSizeY - 2)
			//{
			//	numNeighbors = 5;
			//}
			//if ((i == 1 && j == 1) || (i == 1 && j == paddedSizeY - 2) || (i == paddedSizeX - 2 && j == 1) || (i == paddedSizeX - 2 && j == paddedSizeY - 2))
			//{
			//	numNeighbors = 3;
			//}
			//sum += matrix[i - 1][j - 1];
			//sum += matrix[i - 1][j];
			//sum += matrix[i - 1][j + 1];
			//sum += matrix[i][j - 1];
			//sum += matrix[i][j + 1];
			//sum += matrix[i + 1][j - 1];
			//sum += matrix[i + 1][j];
			//sum += matrix[i + 1][j + 1];
			//resultsMatrix[i - 1][j - 1] = sum / numNeighbors;
	}
}

void firstTest()
{

	//helloFromGpu << <2, 2 >> > ();
	//cudaDeviceSynchronize();

	const int arrSize = 20;
	const int bytesCount = arrSize * sizeof(float);

	float arrIn[arrSize] = {
		1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
		11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f
	};
	float ParrIn = *arrIn;
	float* arrOut = new float[arrSize];

	float* dArrIn;
	float* dArrOut;

	cudaMalloc((void**)&dArrIn, bytesCount);
	cudaMalloc((void**)&dArrOut, bytesCount);


	cudaMemcpy(dArrIn, arrIn, bytesCount, cudaMemcpyHostToDevice);

	square << <1, arrSize >> > (dArrOut, dArrIn);

	cudaDeviceSynchronize();

	cudaMemcpy(arrOut, dArrOut, bytesCount, cudaMemcpyDeviceToHost);

	for (int i = 0; i < arrSize; i++)
	{
		std::cout << arrOut[i] << " ";
	}

	cudaFree(dArrIn);
	cudaFree(dArrOut);

}

int main()
{
	// firstTest();

		//	//	//	settings	//	//	//	//	//	//

	const int sizeX = 18000;	// cols
	const int sizeY = 18000;	// rows
	const int subMatrixesCount = 500;

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


	std::cout << "Otwieranie pliku..." << std::endl;
	CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(matrix, matrixFilePath, sizeY, sizeX);

	double*** subMatrixes = CPC::Common::Helpers::MatrixHelper::divideMatrixToZeroPadded(matrix, sizeX, sizeY, sizeXDivided, sizeY, subMatrixesCount, overlap, lastOverlap);

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

	int dSubMatrixesBytesCount = subMatrixesCount * paddedSizeX * paddedSizeY * sizeof(double);
	int dResultsToMergeBytesCount = subMatrixesCount * sizeY * sizeXDivided * sizeof(double);

	double* dSubMatrixes;
	double* dResultsToMerge;
	cudaMalloc((void**)&dSubMatrixes, dSubMatrixesBytesCount);
	cudaMalloc((void**)&dResultsToMerge, dResultsToMergeBytesCount);

	cudaMemcpy(dSubMatrixes, subMatrixes, dSubMatrixesBytesCount, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(paddedSizeX / threadsPerBlock.x + 1, paddedSizeY / threadsPerBlock.y + 1);

	clock_t start2 = clock();

	//calculateOnPaddedMatrix << <numBlocks, threadsPerBlock >> > ( // );

	cudaMemcpy(allocatedMemoryForResultToMerge, dResultsToMerge, dResultsToMergeBytesCount, cudaMemcpyDeviceToHost);

	double** results2 = CPC::Common::Helpers::MatrixHelper::mergeMatrices(resultsToMerge, sizeX, sizeY, subMatrixesCount, sizeXDivided, overlap, lastOverlap);
	clock_t end2 = clock();

	double duration2 = double(end2 - start2) / CLOCKS_PER_SEC * 1000;
	std::cout << "Obliczenia zakonczono w czasie " << duration2 << " ms" << std::endl;

	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, results2, sizeY, sizeX);

	cudaFree(dResultsToMerge);
	cudaFree(dSubMatrixes);

	return 0;
}
