#include "header_files/concurrent_threads.h"

void watki_wspolbiezne(double *A, double *L, double *U, int n) {

    int i, j, k, m;

    for (k = 0; k < n; k++) {

        #pragma omp parallel for private(m)
        for (j = k; j < n; j++) {
            double sum = 0.0;
            for (m = 0; m < k; m++) {
                sum += L[k*n + m] * U[m*n + j];
            }
            U[k*n + j] = A[k*n + j] - sum;
        }

        #pragma omp barrier

        #pragma omp parallel for private(m)
        for (i = k + 1; i < n; i++) {
            double sum = 0.0;
            for (m = 0; m < k; m++) {
                sum += L[i*n + m] * U[m*n + k];
            }
            L[i*n + k] = (A[i*n + k] - sum) / U[k*n + k];
        }

        #pragma omp barrier
    }
}