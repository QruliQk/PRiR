# Plik file kompiluje cały projekt
# Program makefile sprawdza czy urządzenie ma zainstalowane nvcc
# Jeśli wykryje nvcc -> kompiluje gpgpu.cu
# Jeśli nie ma nvcc  -> kompiluje gpgpu.c 
# =========================

CC = gcc
CFLAGS = -O2 -Wall -Wextra -pedantic -fopenmp
LIBS = -lm

# pliki źródłowe do głównego programu:
MAIN_SRC = PRiR_Projekt1.c concurrent_threads.c komunikaty.c

# plik procesu:
PROC_SRC = lu_proc.c

# sprawdzamy czy jest nvcc (CUDA) 
NVCC := $(shell command -v nvcc 2>/dev/null)

# domyślnie: brak CUDA
GPGPU_OBJ = gpgpu_cpu.o
CUDA_LIBS =

# jeśli nvcc istnieje -> włącz CUDA
ifneq ($(NVCC),)
GPGPU_OBJ = gpgpu.o
CUDA_LIBS = -lcudart
endif

# cele domyślne
all: info PRiR_Projekt1 lu_proc

# krótka informacja co wybrało (CUDA / brak CUDA)
info:
	@echo "== Build info =="
	@if [ -n "$(NVCC)" ]; then echo "CUDA: TAK (nvcc znalezione)"; else echo "CUDA: NIE (brak nvcc)"; fi

# główny program
PRiR_Projekt1: $(MAIN_SRC) $(GPGPU_OBJ)
	$(CC) $(CFLAGS) $(MAIN_SRC) $(GPGPU_OBJ) $(LIBS) $(CUDA_LIBS) -o PRiR_Projekt1

# program do procesów
lu_proc: $(PROC_SRC)
	$(CC) -std=c11 -O2 -Wall -Wextra -pedantic -pthread $(PROC_SRC) -lm -o lu_proc

# CUDA: kompilacja gpgpu.cu -> gpgpu.o
gpgpu.o: gpgpu.cu header_files/gpgpu.h
	nvcc -O2 -c gpgpu.cu -o gpgpu.o

# brak CUDA: kompilacja gpgpu.c -> gpgpu_cpu.o
gpgpu_cpu.o: gpgpu.c header_files/gpgpu.h
	$(CC) -O2 -c gpgpu.c -o gpgpu_cpu.o

# sprzątanie
clean:
	rm -f PRiR_Projekt1 lu_proc *.o
