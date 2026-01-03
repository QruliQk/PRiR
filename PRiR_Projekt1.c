/*
 * Programowanie Równoległe i Rozproszone
 * Projekt: Zadanie 11 – Faktoryzacja LU (Doolittle)
 *
 *  1. watki_wspolbiezne()     -> Adrian Cieśla
 *  2. procesy_wspolbiezne()   -> Bartłomiej Papis
 *  3. komunikaty()            -> Bartosz Wużyński
 *  4. GPGPU()                 -> Mateusz Król
 *
 * Program działa w pętli do momentu wybrania opcji wyjścia.
 * Po każdym wykonaniu można wybrać inny tryb i inny rozmiar macierzy.
 */

/* ======================= BIBLIOTEKI ======================= */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* ======================= PROTOTYPY FUNKCJI ======================= */

/* WERSJA SEKWENCYJNA */
void sekwencyjna(double *A, double *L, double *U, int n);

/* WERSJE RÓWNOLEGŁE */
void watki_wspolbiezne(double *A, double *L, double *U, int n);
void procesy_wspolbiezne(double *A, double *L, double *U, int n);
void komunikaty(double *A, double *L, double *U, int n);
void GPGPU(double *A, double *L, double *U, int n);

/* FUNKCJE POMOCNICZE */
double *allocate_matrix(int n);
void generate_matrix(double *A, int n);
void init_LU(double *L, double *U, int n);
double frobenius_norm(double *A, double *L, double *U, int n);
double get_time(void);
void print_matrix(const char *name, double *M, int n);

/* ======================= MAIN ======================= */

int main(void)
{
    while (1) {

        int n;
        int tryb;

        printf("\n================ MENU ================\n");
        printf("0 - Zakoncz program\n");
        printf("1 - Sekwencyjna\n");
        printf("2 - Watki wspolbiezne\n");
        printf("3 - Procesy wspolbiezne\n");
        printf("4 - Komunikaty (MPI)\n");
        printf("5 - GPGPU\n");
        printf("=====================================\n");
        printf("Twoj wybor: ");
        scanf("%d", &tryb);

        if (tryb == 0) {
            printf("Koniec programu.\n");
            break;
        }

        printf("Podaj rozmiar macierzy n: ");
        scanf("%d", &n);

        double *A = allocate_matrix(n);
        double *L = allocate_matrix(n);
        double *U = allocate_matrix(n);

        if (!A || !L || !U) {
            printf("Blad alokacji pamieci\n");
            return 1;
        }

        generate_matrix(A, n);
        init_LU(L, U, n);

        double t_start = get_time();

        switch (tryb) {
            case 1: sekwencyjna(A, L, U, n); break;
            case 2: watki_wspolbiezne(A, L, U, n); break;
            case 3: procesy_wspolbiezne(A, L, U, n); break;
            case 4: komunikaty(A, L, U, n); break;
            case 5: GPGPU(A, L, U, n); break;
            default:
                printf("Nieznany tryb\n");
                free(A); free(L); free(U);
                continue;
        }

        double t_end = get_time();

        printf("\nCzas wykonania: %.6f s\n", t_end - t_start);
        printf("||A - L*U||_F = %e\n", frobenius_norm(A, L, U, n));

        /* Maksymalny rozmiar macierzy to 8, konsola sobie nie radzi z wiekszymi */
        if (n <= 8) {
            print_matrix("A", A, n);
            print_matrix("L", L, n);
            print_matrix("U", U, n);
        }

        free(A);
        free(L);
        free(U);
    }

    return 0;
}

/* ======================= IMPLEMENTACJE ======================= */

double *allocate_matrix(int n)
{
    return (double *)malloc(n * n * sizeof(double));
}

/* Losowa macierz A + wzmocniona przekątna */
void generate_matrix(double *A, int n)
{
    srand(time(NULL));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = (double)rand() / RAND_MAX;
        }
        A[i*n + i] += n;
    }
}

/* Inicjalizacja L i U */
void init_LU(double *L, double *U, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i*n + j] = (i == j) ? 1.0 : 0.0;
            U[i*n + j] = 0.0;
        }
    }
}

/* Norma Frobeniusa */
double frobenius_norm(double *A, double *L, double *U, int n)
{
    double norm = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += L[i*n + k] * U[k*n + j];
            }

            double diff = A[i*n + j] - sum;
            norm += diff * diff;
        }
    }

    return sqrt(norm);
}

/* Pomiar czasu */
double get_time(void)
{
    struct timespec ts;
    clock_gettime(1, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;

}

/* Wypisywanie macierzy (debug) */
void print_matrix(const char *name, double *M, int n)
{
    printf("\n%s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", M[i*n + j]);
        }
        printf("\n");
    }
}

/* ======== SEKWENCYJNA LU (DOOLITTLE) ======== */
void sekwencyjna(double *A, double *L, double *U, int n)
{
    for (int k = 0; k < n; k++) {

        for (int j = k; j < n; j++) {
            double sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += L[k*n + m] * U[m*n + j];
            }
            U[k*n + j] = A[k*n + j] - sum;
        }

        for (int i = k + 1; i < n; i++) {
            double sum = 0.0;
            for (int m = 0; m < k; m++) {
                sum += L[i*n + m] * U[m*n + k];
            }
            L[i*n + k] = (A[i*n + k] - sum) / U[k*n + k];
        }
    }
}

/* ======== TU WRZUCACIE SWOJĄ CZĘŚĆ PROJEKTU ======== */

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

void procesy_wspolbiezne(double *A, double *L, double *U, int n)
{
}

void komunikaty(double *A, double *L, double *U, int n)
{
}

void GPGPU(double *A, double *L, double *U, int n)
{
}
