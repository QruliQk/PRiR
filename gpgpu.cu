/*
 * Programowanie Równoległe i Rozproszone
 * Projekt PRiR – Zadanie 11
 * Faktoryzacja LU (Doolittle) – GPGPU (CUDA)
 * Autor: Mateusz Król
 */

#include <cuda_runtime.h>
#include <cstdio>

/* Sprawdzania błędów CUDA */
#define CUDA_CHECK(call)                                     \
    do {                                                     \
        cudaError_t err = call;                              \
        if (err != cudaSuccess) {                            \
            fprintf(stderr,                                  \
                "CUDA error %s:%d: %s\n",                    \
                __FILE__, __LINE__,                          \
                cudaGetErrorString(err));                    \
            return;                                          \
        }                                                    \
    } while (0)

/* KERNEL: wiersz U[k][j] */
__global__
void kernel_U_row(const double* A,
                  const double* L,
                  double* U,
                  int n,
                  int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= k && j < n) {
        double sum = 0.0;
        for (int m = 0; m < k; m++) {
            sum += L[k * n + m] * U[m * n + j];
        }
        U[k * n + j] = A[k * n + j] - sum;
    }
}

/* KERNEL: kolumna L[i][k] */
__global__
void kernel_L_col(const double* A,
                  double* L,
                  const double* U,
                  int n,
                  int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > k && i < n) {
        double sum = 0.0;
        for (int m = 0; m < k; m++) {
            sum += L[i * n + m] * U[m * n + k];
        }
        L[i * n + k] = (A[i * n + k] - sum) / U[k * n + k];
    }
}

/* Funkcja wywoływana z PRiR_Projekt1.c */
extern "C"
void GPGPU(double* A, double* L, double* U, int n)
{
    /* Sprawdzenie, czy dostępne jest urządzenie CUDA */
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "[GPGPU] Brak urzadzen CUDA. Tryb GPGPU niedostepny.\n");
        return;
    }

    size_t bytes = n * n * sizeof(double);

    double* d_A = nullptr;
    double* d_L = nullptr;
    double* d_U = nullptr;

    /* Alokacja GPU */
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_L, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_U, bytes));

    /* Kopiowanie CPU -> GPU */
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_L, L, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_U, U, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    /* Pełna faktoryzacja LU */
    for (int k = 0; k < n; k++) {

        kernel_U_row<<<blocks, threads>>>(d_A, d_L, d_U, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        kernel_L_col<<<blocks, threads>>>(d_A, d_L, d_U, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* Kopiowanie GPU -> CPU */
    CUDA_CHECK(cudaMemcpy(U, d_U, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(L, d_L, bytes, cudaMemcpyDeviceToHost));

    /* Zwolnienie GPU */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_U));
}
