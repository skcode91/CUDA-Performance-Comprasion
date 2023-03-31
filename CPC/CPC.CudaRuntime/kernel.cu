
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include "../CPC.Common/Helpers/MatrixHelper.cpp"
#include "../CPC.Common/Helpers/CalculationHelper.h"
#include "../CPC.Common/Helpers/BinaryFileHelper.cpp"

void printMatrix(double* mat, int sizeX, int sizeY) {
	for (int i = 0; i < sizeX; ++i) {
		for (int j = 0; j < sizeY; ++j) {
			std::cout << mat[i * sizeX + j] << " ";
		}
		std::cout << std::endl;
	}
}

__global__ void printGpuMatrix(double* mat, int sizeX, int sizeY) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int i = 0; i < sizeX; ++i) {
			for (int j = 0; j < sizeY; ++j) {
				printf("%f ", mat[i * sizeX + j]);
			}
			printf("\n");
		}
	}
}

__global__ void testMatrix(double* matrix, const int sizeX, const int sizeY)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int stride;


	printf("\n%d %d %f", i, j);
}

__global__ void calculateOnPaddedMatrix(double* input, double* output, const int paddedSizeX, const int paddedSizeY, const int subMatrixesCount)
{
	int dimension = blockDim.x;
	int block = blockIdx.x;
	int thread = threadIdx.x;

	int inputOffset = (block * dimension + thread) * paddedSizeX * paddedSizeY;
	int outputOffset = (block * dimension + thread) * (paddedSizeX - 2) * (paddedSizeY - 2);
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

__global__ void mergeMatrices(double* allocatedData, double*** PResultsToMerge, int sizeX, int sizeY, int subMatrixesCount, int sizeXDivided, int overlap, int lastOverlap)
{
	int dimension = blockDim.x;
	int block = blockIdx.x;
	int thread = threadIdx.x;

	int index = block * dimension + thread;

	if (index < sizeX * sizeY) {
		int x = index / sizeY;
		int y = index % sizeY;

		int i = x / sizeXDivided;
		if (i < subMatrixesCount - 1) {
			int xOffset = x * sizeY;
			int yOffset = y;
			int absoluteOffset = i * (sizeXDivided * sizeY - (overlap * sizeY));
			int offset = absoluteOffset + xOffset + yOffset;

			if (i > 0 && x % sizeXDivided == 0) {
				return;
			}

			allocatedData[offset] = PResultsToMerge[i][x % sizeXDivided][y];
		}
		else if (i == subMatrixesCount - 1 && x >= sizeX - sizeXDivided) {
			int xOffset = x * sizeY;
			int yOffset = y;
			int absoluteOffset = (sizeX - sizeXDivided) * sizeY;
			int offset = absoluteOffset + xOffset + yOffset;

			allocatedData[offset] = PResultsToMerge[subMatrixesCount - 1][x % sizeXDivided][y];
		}
	}
}

__global__ void printKernel(double* gpuOutput) {
	for (int i = 0; i < 100; i++) {
		printf("gpuOutput[%d]: %f\n", i, gpuOutput[i]);
	}
}

__global__ void subMatrixOnGpu2d(double* matrix, const int sizeX, const int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;

	if (ix < sizeX && iy < sizeY)
		printf("\n ix %d iy %d", ix, iy);
}

__device__ inline double getMedian(double* values, int size)
{
	for (int i = 1; i < size; i++) {
		double key = values[i];
		int j = i - 1;
		while (j >= 0 && values[j] > key) {
			values[j + 1] = values[j];
			j = j - 1;
		}
		values[j + 1] = key;
	}
	if (size % 2 == 0) {
		return (values[size / 2 - 1] + values[size / 2]) / 2.0;
	}
	else {
		return values[size / 2];
	}
}

__global__ void medianFilter(double* inputMatrix, double* outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;

	if (ix >= 1 && iy >= 1 && ix < sizeX - 1 && iy < sizeY - 1) {
		double windowValues[9];
		int windowIndex = 0;
		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = -1; dy <= 1; dy++) {
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 9);
	}
}

__global__ void medianFilterFirstRow(double* inputMatrix, double* outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
	unsigned int iy = 0;
	unsigned int idx = iy * sizeX + ix;

	if (ix < sizeX - 1) {
		double windowValues[6];
		int windowIndex = 0;
		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = 0; dy <= 1; dy++) {
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 6);
	}
}

__global__ void medianFilterFirstColumn(double* inputMatrix, double* outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = 0;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
	unsigned int idx = iy * sizeX + ix;

	if (iy < sizeY - 1) {
		double windowValues[6];
		int windowIndex = 0;
		for (int dx = 0; dx <= 1; dx++) {
			for (int dy = -1; dy <= 1; dy++) {
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 6);
	}
}

__global__ void medianFilterLastColumn(double* inputMatrix, double* outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = sizeX - 1;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y + 1;
	unsigned int idx = iy * sizeX + ix;

	if (iy < sizeY - 1) {
		double windowValues[6];
		int windowIndex = 0;
		for (int dx = -1; dx <= 0; dx++) {
			for (int dy = -1; dy <= 1; dy++) {
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 6);
	}
}


__global__ void medianFilterLastRow(double* inputMatrix, double* outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
	unsigned int iy = sizeY - 1;
	unsigned int idx = iy * sizeX + ix;

	if (ix < sizeX - 1) {
		double windowValues[6];
		int windowIndex = 0;
		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = -1; dy <= 0; dy++) {
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 6);
	}
}

__global__ void calculateCorners(double* inputMatrix, double* outputMatrix, int sizeX, int sizeY)
{
	double cornerValues[4];


	// left - up corner
	cornerValues[0] = inputMatrix[0];
	cornerValues[1] = inputMatrix[1];
	cornerValues[2] = inputMatrix[sizeX];
	cornerValues[3] = inputMatrix[sizeX + 1];
	outputMatrix[0] = getMedian(cornerValues, 4);

	// right - up corner
	cornerValues[0] = inputMatrix[sizeX - 2];
	cornerValues[1] = inputMatrix[sizeX - 1];
	cornerValues[2] = inputMatrix[2 * sizeX - 2];
	cornerValues[3] = inputMatrix[2 * sizeX - 1];
	outputMatrix[sizeX - 1] = getMedian(cornerValues, 4);

	// left - down corner
	cornerValues[0] = inputMatrix[(sizeY - 2) * sizeX];
	cornerValues[1] = inputMatrix[(sizeY - 2) * sizeX + 1];
	cornerValues[2] = inputMatrix[(sizeY - 1) * sizeX];
	cornerValues[3] = inputMatrix[(sizeY - 1) * sizeX + 1];
	outputMatrix[(sizeY - 1) * sizeX] = getMedian(cornerValues, 4);

	// right - down corner
	cornerValues[0] = inputMatrix[sizeX * (sizeY - 1) - 2];
	cornerValues[1] = inputMatrix[sizeX * (sizeY - 1) - 1];
	cornerValues[2] = inputMatrix[sizeX * sizeY - 2];
	cornerValues[3] = inputMatrix[sizeX * sizeY - 1];
	outputMatrix[sizeX * sizeY - 1] = getMedian(cornerValues, 4);
}

int main()
{
		//	//	//	settings	//	//	//	//	//	//

	const int sizeX = 19000;	// cols
	const int sizeY = 19000;	// rows

	const std::string matrixFilePath = "D:/matrix.bin";
	const std::string probeFilePath = "D:/probe.txt";

	//	//	//	//	//	//	//	//	//	//	//	//

	std::cout << "Ilosc pamieci wykorzystywanej przez macierz to " << sizeY * sizeX / 1024 / 1024 * sizeof(double) << " Mb" << std::endl;

	// allocate memory

	double* inputMatrixVector = new double[sizeX * sizeY];
	double** inputMatrix = new double* [sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		inputMatrix[i] = inputMatrixVector + i * sizeY;
	}

	double* outputMatrixVector = new double[sizeX * sizeY];
	double** outputMatrix = new double* [sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		outputMatrix[i] = outputMatrixVector + i * sizeY;
	}


	std::cout << "Otwieranie pliku..." << std::endl;
	CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(inputMatrix, matrixFilePath, sizeY, sizeX);


	// allocate memory on GPU
	double* gpuOutputMatrix;
	double* gpuInputMatrix;


	cudaMalloc(&gpuInputMatrix, sizeX * sizeY * sizeof(double));
	cudaMalloc(&gpuOutputMatrix, sizeX * sizeY * sizeof(double));

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("cudaMemcpy error 1: %s\n", cudaGetErrorString(error));
	}
	// copy from host to gpu
	cudaMemcpy(gpuInputMatrix, inputMatrixVector, sizeX * sizeY * sizeof(double), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("cudaMemcpy error 1: %s\n", cudaGetErrorString(error));
	}

	// kernel sizes
	int blockSizeX = 32;
	int blockSizeY = 32;


	dim3 blockMedian(blockSizeX, blockSizeY);
	dim3 gridMedian((sizeX + blockMedian.x - 1) / blockMedian.x, (sizeY + blockMedian.y - 1) / blockMedian.y);

	dim3 blockFirstRow(blockSizeX);
	dim3 gridFirstRow((sizeX - 2 + blockFirstRow.x - 1) / blockFirstRow.x);

	dim3 blockFirstColumn(1, blockSizeY);
	dim3 gridFirstColumn((sizeY - 2 + blockFirstColumn.y - 1) / blockFirstColumn.y);

	dim3 blockLastColumn(1, blockSizeY);
	dim3 gridLastColumn((sizeY - 2 + blockLastColumn.y - 1) / blockLastColumn.y);

	dim3 blockLastRow(blockSizeX, 1);
	dim3 gridLastRow((sizeX - 2 + blockLastRow.x - 1) / blockLastRow.x);

	// computing 
	clock_t start = clock();

	medianFilter << <gridMedian, blockMedian >> > (gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);

	cudaDeviceSynchronize();

	medianFilterFirstRow << <gridFirstRow, blockFirstRow >> > (gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
	medianFilterLastRow << <gridLastRow, blockLastRow >> > (gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
	medianFilterFirstColumn << <gridFirstColumn, blockFirstColumn >> > (gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
	medianFilterLastColumn << <gridLastColumn, blockLastColumn >> > (gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
	calculateCorners << <1, 1 >> > (gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);

	cudaDeviceSynchronize();

	clock_t end = clock();

	// end of computing


	double duration = double(end - start) / CLOCKS_PER_SEC * 500;

	std::cout << "Obliczenia zakonczono w czasie " << duration << " ms" << std::endl;

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("cudaMemcpy error 4: %s\n", cudaGetErrorString(error));
	}

	cudaDeviceSynchronize();

	cudaMemcpy(outputMatrixVector, gpuOutputMatrix, sizeX * sizeY * sizeof(double), cudaMemcpyDeviceToHost);


	//std::cout << std::endl;
	//printMatrix(outputMatrix[0], sizeX, sizeY);
	//std::cout << std::endl;
	//printMatrix(inputMatrix[0], sizeX, sizeY);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("cudaMemcpy error (DeviceToHost): %s\n", cudaGetErrorString(error));
	}
	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, outputMatrix, sizeY, sizeX);

	cudaFree(gpuInputMatrix);
	cudaFree(gpuOutputMatrix);

	return 0;
}
