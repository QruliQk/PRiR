#include <stdio.h>
#include "header_files/gpgpu.h"

/* Brak CUDA: wersja CPU algorytmu LU  */
void GPGPU(double* A, double* L, double* U, int n)
{
    fprintf(stderr,
        "\nGPGPU niedostepne: brak dzialajacego CUDA device.\n"
        "Brak CUDA: wersja CPU algorytmu LU. \n\n");

    for (int k = 0; k < n; k++) {

        /* wiersz U[k][j] */
        for (int j = k; j < n; j++) {
            double sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += L[k * n + m] * U[m * n + j];
            }
            U[k * n + j] = A[k * n + j] - sum;
        }

        /* kolumna L[i][k] */
        for (int i = k + 1; i < n; i++) {
            double sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += L[i * n + m] * U[m * n + k];
            }
            L[i * n + k] = (A[i * n + k] - sum) / U[k * n + k];
        }
    }
}
