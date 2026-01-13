# PRiR Project
## Compile a program:
```
nvcc -O2 -c gpgpu.cu -o gpgpu.o
```
```
gcc -fopenmp PRiR_Projekt1.c concurrent_threads.c gpgpu.o -lm -o PRiR_Projekt1
```
```
gcc -std=c11 -O2 -Wall -Wextra -pedantic -pthread lu_proc.c -lm -o lu_proc
```
## Run a program:
```
./PRiR_Projekt1
```
