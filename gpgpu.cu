/*
 * Programowanie Równoległe i Rozproszone
 * Projekt PRiR – Zadanie 11
 * Faktoryzacja LU (Doolittle) – GPGPU (CUDA)
 *
 * Autor: Mateusz Król
 */

#include <cuda_runtime.h>
#include <cstdio>

 /* =========================================================
    KERNEL: wiersz U[k][j]
    ========================================================= */
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

/* =========================================================
   KERNEL: kolumna L[i][k]
   ========================================================= */
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

/* =========================================================
   FUNKCJA WYWOŁYWANA Z main.c
   ========================================================= */
extern "C"
void GPGPU(double* A, double* L, double* U, int n)
{
    size_t bytes = n * n * sizeof(double);

    double* d_A = nullptr;
    double* d_L = nullptr;
    double* d_U = nullptr;

    /* Alokacja GPU */
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_L, bytes);
    cudaMalloc((void**)&d_U, bytes);

    /* Kopiowanie CPU -> GPU */
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    /* ===== PEŁNA FAKTORYZACJA LU ===== */
    for (int k = 0; k < n; k++) {

        kernel_U_row << <blocks, threads >> > (d_A, d_L, d_U, n, k);
        cudaDeviceSynchronize();

        kernel_L_col << <blocks, threads >> > (d_A, d_L, d_U, n, k);
        cudaDeviceSynchronize();
    }

    /* Kopiowanie GPU -> CPU */
    cudaMemcpy(U, d_U, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, bytes, cudaMemcpyDeviceToHost);

    /* Zwolnienie GPU */
    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);
}
