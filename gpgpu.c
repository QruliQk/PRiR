#include <stdio.h>
#include "header_files/gpgpu.h"

void GPGPU(double* A, double* L, double* U, int n)
{
    (void)A; (void)L; (void)U; (void)n;

    printf("\n[GPGPU] Niedostepne: brak CUDA (nvcc) lub brak kompatybilnego GPU NVIDIA.\n");
    printf("[GPGPU] Aby wlaczyc GPGPU: zainstaluj CUDA Toolkit i skompiluj na maszynie z NVIDIA.\n\n");
}
