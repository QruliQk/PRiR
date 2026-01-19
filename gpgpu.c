#include <stdio.h>
#include "header_files/gpgpu.h"

void GPGPU(double* A, double* L, double* U, int n)
{
    (void)A; (void)L; (void)U; (void)n;

    printf("\nGPGPU Niedostepne: brak CUDA (nvcc) lub brak kompatybilnego GPU NVIDIA.\n");
    printf("Aby wlaczyc GPGPU zainstaluj CUDA Toolkit i skompiluj na maszynie z NVIDIA.\n\n");
}

