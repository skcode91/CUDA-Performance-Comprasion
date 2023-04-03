
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include "../CPC.Common/Helpers/MatrixHelper.h"
#include "../CPC.Common/Helpers/BinaryFileHelper.h"

__device__ inline double getMedian(double *values, int size)
{
	for (int i = 1; i < size; i++)
	{
		double key = values[i];
		int j = i - 1;
		while (j >= 0 && values[j] > key)
		{
			values[j + 1] = values[j];
			j = j - 1;
		}
		values[j + 1] = key;
	}
	if (size % 2 == 0)
	{
		return (values[size / 2 - 1] + values[size / 2]) / 2.0;
	}
	else
	{
		return values[size / 2];
	}
}

__global__ void medianFilter(double *inputMatrix, double *outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * sizeX + ix;

	if (ix >= 1 && iy >= 1 && ix < sizeX - 1 && iy < sizeY - 1)
	{
		double windowValues[9];
		int windowIndex = 0;
		for (int dx = -1; dx <= 1; dx++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 9);
	}
}

__global__ void medianFilterFirstRow(double *inputMatrix, double *outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
	unsigned int iy = 0;
	unsigned int idx = iy * sizeX + ix;

	if (ix < sizeX - 1)
	{
		double windowValues[6];
		int windowIndex = 0;
		for (int dx = -1; dx <= 1; dx++)
		{
			for (int dy = 0; dy <= 1; dy++)
			{
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 6);
	}
}

__global__ void medianFilterFirstColumn(double *inputMatrix, double *outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = 0;
	unsigned int iy = threadIdx.x + blockIdx.x * blockDim.x + 1;
	unsigned int idx = iy * sizeX + ix;

	if (iy < sizeY - 1)
	{
		double windowValues[6];
		int windowIndex = 0;
		for (int dx = 0; dx <= 1; dx++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 6);
	}
}

__global__ void medianFilterLastColumn(double *inputMatrix, double *outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = sizeX - 1;
	unsigned int iy = threadIdx.x + blockIdx.x * blockDim.x + 1;
	unsigned int idx = iy * sizeX + ix;

	if (iy < sizeY - 1)
	{
		double windowValues[6];
		int windowIndex = 0;
		for (int dx = -1; dx <= 0; dx++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 6);
	}
}

__global__ void medianFilterLastRow(double *inputMatrix, double *outputMatrix, int sizeX, int sizeY)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x + 1;
	unsigned int iy = sizeY - 1;
	unsigned int idx = iy * sizeX + ix;

	if (ix < sizeX - 1)
	{
		double windowValues[6];
		int windowIndex = 0;
		for (int dx = -1; dx <= 1; dx++)
		{
			for (int dy = -1; dy <= 0; dy++)
			{
				unsigned int windowIdx = (iy + dy) * sizeX + (ix + dx);
				windowValues[windowIndex++] = inputMatrix[windowIdx];
			}
		}
		outputMatrix[idx] = getMedian(windowValues, 6);
	}
}

__global__ void calculateCorners(double *inputMatrix, double *outputMatrix, int sizeX, int sizeY)
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

	const int sizeX = 19000; // cols
	const int sizeY = 19000; // rows
	int cycles = 5;

	const std::string matrixFilePath = "d:/matrix.bin";
	const std::string probeFilePath = "d:/probe.txt";

	//	//	//	//	//	//	//	//	//	//	//	//

	std::cout << "allocated memory: 2 x " << sizeY * sizeX / 1024 / 1024 * sizeof(double) << " Mb, matrix size: " << sizeX << " x " << sizeY << std::endl;

	// allocate memory

	double *inputMatrixVector = new double[sizeX * sizeY];
	double **inputMatrix = new double *[sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		inputMatrix[i] = inputMatrixVector + i * sizeY;
	}

	double *outputMatrixVector = new double[sizeX * sizeY];
	double **outputMatrix = new double *[sizeX];
	for (int i = 0; i < sizeX; i++)
	{
		outputMatrix[i] = outputMatrixVector + i * sizeY;
	}


	// allocate memory on GPU
	double *gpuOutputMatrix;
	double *gpuInputMatrix;

	cudaMalloc(&gpuInputMatrix, sizeX * sizeY * sizeof(double));
	cudaMalloc(&gpuOutputMatrix, sizeX * sizeY * sizeof(double));

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("error: %s\n", cudaGetErrorString(error));
	}

	//reading data on host
	std::cout << "opening file..." << std::endl;
	CPC::Common::Helpers::BinaryFileHelper::readMatrixFromFile(inputMatrix, matrixFilePath, sizeY, sizeX);

	// copy from host to gpu
	cudaMemcpy(gpuInputMatrix, inputMatrixVector, sizeX * sizeY * sizeof(double), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("error: %s\n", cudaGetErrorString(error));
	}

	// kernel sizes
	int blockSizeX = 32;
	int blockSizeY = 32;

	dim3 blockMedian(blockSizeX, blockSizeY);
	dim3 gridMedian((sizeX + blockMedian.x - 1) / blockMedian.x, (sizeY + blockMedian.y - 1) / blockMedian.y);

	dim3 blockFirstRow(blockSizeX);
	dim3 gridFirstRow((sizeX - 2 + blockFirstRow.x - 1) / blockFirstRow.x);

	dim3 blockFirstColumn(blockSizeX);
	dim3 gridFirstColumn((sizeX - 2 + blockFirstColumn.x - 1) / blockFirstColumn.x);

	dim3 blockLastColumn(blockSizeX);
	dim3 gridLastColumn((sizeX - 2 + blockLastColumn.x - 1) / blockLastColumn.x);

	dim3 blockLastRow(blockSizeX, 1);
	dim3 gridLastRow((sizeX - 2 + blockLastRow.x - 1) / blockLastRow.x);

	// create cuda timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start the timer
	cudaEventRecord(start, 0);

	// computing
	for (int i = 0; i < cycles; i++)
	{
		std::cout << "computing iteration " << i + 1 << "/" << cycles << std::endl;

		if (i % 2 == 0)
		{
			medianFilter<<<gridMedian, blockMedian>>>(gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
			cudaDeviceSynchronize();
			medianFilterFirstRow<<<gridFirstRow, blockFirstRow>>>(gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
			medianFilterLastRow<<<gridLastRow, blockLastRow>>>(gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
			medianFilterFirstColumn<<<gridFirstColumn, blockFirstColumn>>>(gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
			medianFilterLastColumn<<<gridLastColumn, blockLastColumn>>>(gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
			calculateCorners<<<1, 1>>>(gpuInputMatrix, gpuOutputMatrix, sizeX, sizeY);
			cudaDeviceSynchronize();
		}
		else
		{
			medianFilter<<<gridMedian, blockMedian>>>(gpuOutputMatrix, gpuInputMatrix, sizeX, sizeY);
			cudaDeviceSynchronize();
			medianFilterFirstRow<<<gridFirstRow, blockFirstRow>>>(gpuOutputMatrix, gpuInputMatrix, sizeX, sizeY);
			medianFilterLastRow<<<gridLastRow, blockLastRow>>>(gpuOutputMatrix, gpuInputMatrix, sizeX, sizeY);
			medianFilterFirstColumn<<<gridFirstColumn, blockFirstColumn>>>(gpuOutputMatrix, gpuInputMatrix, sizeX, sizeY);
			medianFilterLastColumn<<<gridLastColumn, blockLastColumn>>>(gpuOutputMatrix, gpuInputMatrix, sizeX, sizeY);
			calculateCorners<<<1, 1>>>(gpuOutputMatrix, gpuInputMatrix, sizeX, sizeY);
			cudaDeviceSynchronize();
		}
	}

	// end of computing

	// stop the timer and calculate the elapsed time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);

	// print the result
	std::cout << "computing finished in " << elapsed / 1000.0 << " s" << std::endl;
	
	// cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("error: %s\n", cudaGetErrorString(error));
	}

	cudaDeviceSynchronize();

	cudaMemcpy(outputMatrixVector, cycles % 2 == 0 ? gpuInputMatrix : gpuOutputMatrix, sizeX * sizeY * sizeof(double), cudaMemcpyDeviceToHost);

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("error: %s\n", cudaGetErrorString(error));
	}
	CPC::Common::Helpers::BinaryFileHelper::validateMatrixProbe(probeFilePath, outputMatrix, sizeY, sizeX);

	cudaFree(gpuInputMatrix);
	cudaFree(gpuOutputMatrix);

	return 0;
}
