# Plik file kompiluje cały projekt
# Program makefile sprawdza czy urządzenie ma zainstalowane nvcc
# Jeśli wykryje nvcc -> kompiluje gpgpu.cu
# Jeśli nie ma nvcc  -> kompiluje gpgpu.c 
# =========================

CC      := gcc
NVCC    := nvcc

CFLAGS  := -O2 -Wall -Wextra -pedantic -fopenmp
LDFLAGS := -lm

MAIN_SRC := PRiR_Projekt1.c
CPU_SRCS := concurrent_threads.c komunikaty.c

# Czy jest nvcc
HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)

# tworzymy tymczasowy plik .cu
# kompilujemy nvcc
# uruchamiamy
# sprzatamy
CUDA_OK := 0
ifeq ($(HAVE_NVCC),1)
CUDA_OK := $(shell \
  rm -f .cuda_check .cuda_check.cu >/dev/null 2>&1; \
  printf '%s\n' \
    '#include <cuda_runtime.h>' \
    'int main(){int c=0; cudaError_t e=cudaGetDeviceCount(&c); return (e==cudaSuccess && c>0)?0:1;}' \
    > .cuda_check.cu 2>/dev/null; \
  $(NVCC) -O2 .cuda_check.cu -o .cuda_check >/dev/null 2>&1 && \
  ./.cuda_check >/dev/null 2>&1 && echo 1 || echo 0; \
  rm -f .cuda_check .cuda_check.cu >/dev/null 2>&1 \
)
endif

ifeq ($(CUDA_OK),1)
GPGPU_OBJ := gpgpu.o
GPGPU_MSG := "CUDA: TAK (nvcc + device OK) -> buduje gpgpu.cu"
CUDA_LIBS := -lcudart
else
GPGPU_OBJ := gpgpu_cpu.o
GPGPU_MSG := "CUDA: NIE (brak device lub brak nvcc) -> buduje gpgpu.c"
CUDA_LIBS :=
endif

all: info PRiR_Projekt1 lu_proc

info:
	@echo "== Build info =="
	@echo $(GPGPU_MSG)

PRiR_Projekt1: $(MAIN_SRC) $(CPU_SRCS) $(GPGPU_OBJ)
	$(CC) $(CFLAGS) $(MAIN_SRC) $(CPU_SRCS) $(GPGPU_OBJ) -o $@ $(LDFLAGS) $(CUDA_LIBS)

gpgpu.o: gpgpu.cu
	$(NVCC) -O2 -c $< -o $@

gpgpu_cpu.o: gpgpu.c
	$(CC) $(CFLAGS) -c $< -o $@

lu_proc: lu_proc.c
	$(CC) -std=c11 -O2 -Wall -Wextra -pedantic -pthread $< -o $@ -lm

clean:
	rm -f PRiR_Projekt1 lu_proc *.o .cuda_check .cuda_check.cu
	rm -f PRiR_Projekt1 lu_proc *.o
