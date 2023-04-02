#include <iostream>
#include "./MultiCore.h"
#include "../Helpers/MatrixHelper.h"
#include <omp.h>

namespace CPC
{
    namespace Common
    {
        namespace MultiCore
        {
            namespace MultiCore
            {
                void test()
                {
                    std::cout << "test";
                }

                void medianFilter(double **PInput, double **POutput, int sizeY, int sizeX)
                {

#pragma omp parallel
                    {
                        double *windowArr = new double[9];
#pragma omp for schedule(static, 1)
                        for (int i = 1; i < sizeY - 1; i++)
                        {
                            for (int j = 1; j < sizeX - 1; j++)
                            {
                                windowArr[0] = PInput[i - 1][j - 1];
                                windowArr[1] = PInput[i - 1][j];
                                windowArr[2] = PInput[i - 1][j + 1];
                                windowArr[3] = PInput[i][j - 1];
                                windowArr[4] = PInput[i][j + 1];
                                windowArr[5] = PInput[i + 1][j - 1];
                                windowArr[6] = PInput[i + 1][j];
                                windowArr[7] = PInput[i + 1][j + 1];
                                windowArr[8] = PInput[i][j];

                                POutput[i][j] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 9);
                            }
                        }
                        delete[] windowArr;
                    }

// first and last column
#pragma omp parallel
                    {
                        double *windowArr = new double[9];
#pragma omp for schedule(static, 1)
                        for (int i = 1; i < sizeY - 1; i++)
                        {
                            windowArr[0] = PInput[i - 1][0];
                            windowArr[1] = PInput[i - 1][1];
                            windowArr[2] = PInput[i][1];
                            windowArr[3] = PInput[i + 1][0];
                            windowArr[4] = PInput[i + 1][1];
                            windowArr[5] = PInput[i][0];
                            POutput[i][0] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 6);

                            windowArr[0] = PInput[i - 1][sizeX - 2];
                            windowArr[1] = PInput[i - 1][sizeX - 1];
                            windowArr[2] = PInput[i][sizeX - 2];
                            windowArr[3] = PInput[i + 1][sizeX - 2];
                            windowArr[4] = PInput[i + 1][sizeX - 1];
                            windowArr[5] = PInput[i][sizeX - 1];
                            POutput[i][sizeX - 1] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 6);
                        }
                        delete[] windowArr;
                    }

// first and last row
#pragma omp parallel
                    {
                        double *windowArr = new double[9];
#pragma omp for schedule(static, 1)
                        for (int i = 1; i < sizeX - 1; i++)
                        {
                            windowArr[0] = PInput[0][i - 1];
                            windowArr[1] = PInput[1][i - 1];
                            windowArr[2] = PInput[1][i];
                            windowArr[3] = PInput[0][i + 1];
                            windowArr[4] = PInput[1][i + 1];
                            windowArr[5] = PInput[0][i];

                            POutput[0][i] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 6);

                            windowArr[0] = PInput[sizeY - 2][i - 1];
                            windowArr[1] = PInput[sizeY - 1][i - 1];
                            windowArr[2] = PInput[sizeY - 2][i];
                            windowArr[3] = PInput[sizeY - 2][i + 1];
                            windowArr[4] = PInput[sizeY - 1][i + 1];
                            windowArr[5] = PInput[sizeY - 1][i];
                            POutput[sizeY - 1][i] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 6);
                        }
                        delete[] windowArr;
                    }

                    // corners
                    double *windowArr = new double[4];
                    windowArr[0] = PInput[0][1];
                    windowArr[1] = PInput[1][1];
                    windowArr[2] = PInput[1][0];
                    windowArr[3] = PInput[0][0];
                    POutput[0][0] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 4);

                    windowArr[0] = PInput[0][sizeX - 2];
                    windowArr[1] = PInput[1][sizeX - 2];
                    windowArr[2] = PInput[1][sizeX - 1];
                    windowArr[3] = PInput[0][sizeX - 1];
                    POutput[0][sizeX - 1] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 4);

                    windowArr[0] = PInput[sizeY - 2][0];
                    windowArr[1] = PInput[sizeY - 2][1];
                    windowArr[2] = PInput[sizeY - 1][1];
                    windowArr[3] = PInput[sizeY - 1][0];
                    POutput[sizeY - 1][0] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 4);

                    windowArr[0] = PInput[sizeY - 2][sizeX - 2];
                    windowArr[1] = PInput[sizeY - 2][sizeX - 1];
                    windowArr[2] = PInput[sizeY - 1][sizeX - 2];
                    windowArr[3] = PInput[sizeY - 1][sizeX - 1];
                    POutput[sizeY - 1][sizeX - 1] = CPC::Common::Helpers::MatrixHelper::median(windowArr, 4);

                    delete[] windowArr;
                }
            }
        }
    }
}