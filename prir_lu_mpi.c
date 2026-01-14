// prir_lu_mpi.c
// LU Doolittle (bez pivotowania), MPI (message passing), diag(L)=1
// Kompilacja: mpicc -O2 -std=c11 -Wall -Wextra -pedantic prir_lu_mpi.c -lm -o prir_lu_mpi
// Uruchomienie: mpirun -np 4 ./prir_lu_mpi 2000 123 3
//              mpirun -np 1 ./prir_lu_mpi 2000 123 3   (jako T(n,1) w MPI)

#include <mpi.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die_rank(int rank, const char *msg) {
    fprintf(stderr, "[rank %d] ERROR: %s\n", rank, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static size_t idx(size_t n, size_t i, size_t j) { return i * n + j; }

/* ---------- RNG (xorshift64*) ---------- */
static uint64_t xorshift64s(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}
static double rnd_u11(uint64_t *state) {
    uint64_t r = xorshift64s(state);
    uint64_t mant = r >> 11; // 53 bity
    double u = (double)mant / (double)((1ULL << 53) - 1ULL); // [0,1]
    return 2.0 * u - 1.0; // [-1,1]
}

static void generate_A_diagonally_dominant(size_t n, double *A, uint64_t seed) {
    uint64_t st = seed ? seed : 1ULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A[idx(n, i, j)] = rnd_u11(&st);
        }
    }

    // diagonal dominance: A[i,i] += sum_{j!=i} |A[i,j]| + 1
    for (size_t i = 0; i < n; i++) {
        long double sum = 0.0L;
        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;
            sum += fabsl((long double)A[idx(n, i, j)]);
        }
        A[idx(n, i, i)] += (double)(sum + 1.0L);
    }
}

/* ---------- Normy / poprawność (rank 0, poza czasem) ---------- */
static double frob_norm(size_t n, const double *M) {
    long double acc = 0.0L;
    size_t nn = n * n;
    for (size_t t = 0; t < nn; t++) {
        long double v = (long double)M[t];
        acc += v * v;
    }
    return (double)sqrt((double)acc);
}

static void matmul_LU_minus_A(size_t n, const double *L, const double *U, const double *A, double *Out) {
    // Out = L*U - A ; wykorzystujemy trójkątność by ograniczyć pętlę po k
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            long double sum = 0.0L;
            size_t k0 = (j < i) ? j : i; // L(i,k)=0 dla k>i, U(k,j)=0 dla k>j
            for (size_t k = 0; k <= k0; k++) {
                sum += (long double)L[idx(n, i, k)] * (long double)U[idx(n, k, j)];
            }
            Out[idx(n, i, j)] = (double)sum - A[idx(n, i, j)];
        }
    }
}

/* ---------- Podział zakresu 0..len-1 na p części ---------- */
static void block_range(size_t len, int p, int rank, size_t *start, size_t *end) {
    *start = (size_t)rank * len / (size_t)p;
    *end   = (size_t)(rank + 1) * len / (size_t)p;
}

/* ---------- LU Doolittle MPI ---------- */
static int lu_doolittle_mpi(size_t n, int p, int rank, const double *A, double *L, double *U) {
    // pomocnicze bufory do komunikacji
    int *counts = (int *)calloc((size_t)p, sizeof(int));
    int *displs = (int *)calloc((size_t)p, sizeof(int));
    if (!counts || !displs) die_rank(rank, "calloc counts/displs failed");

    double *local = (double *)malloc(n * sizeof(double)); // max segment
    double *colbuf = (double *)malloc(n * sizeof(double)); // max lenL
    if (!local || !colbuf) die_rank(rank, "malloc local buffers failed");

    const double eps_pivot = 1e-12;

    for (size_t k = 0; k < n; k++) {
        /* --------- Faza U: licz U[k, j] dla j=k..n-1 (dzielimy po j) --------- */
        size_t lenU = n - k;
        size_t sU, eU;
        block_range(lenU, p, rank, &sU, &eU);
        size_t locU = eU - sU;

        // policz lokalny fragment wiersza U (indeksujemy od 0..locU-1)
        for (size_t t = 0; t < locU; t++) {
            size_t j = k + (sU + t);
            long double sum = 0.0L;
            for (size_t m = 0; m < k; m++) {
                sum += (long double)L[idx(n, k, m)] * (long double)U[idx(n, m, j)];
            }
            local[t] = A[idx(n, k, j)] - (double)sum;
        }

        // zbuduj counts/displs dla Allgatherv (w obrębie segmentu długości lenU)
        for (int r = 0; r < p; r++) {
            size_t sr, er;
            block_range(lenU, p, r, &sr, &er);
            counts[r] = (int)(er - sr);
            displs[r] = (int)sr;
        }

        // zbierz cały wiersz U[k, k..n-1] u wszystkich
        double *Urow = U + idx(n, k, k); // lenU elementów ciągłych
        MPI_Allgatherv(local, (int)locU, MPI_DOUBLE,
                       Urow, counts, displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // pivot check
        double pivot = U[idx(n, k, k)];
        int bad = (fabs(pivot) < eps_pivot) ? 1 : 0;
        int bad_global = 0;
        MPI_Allreduce(&bad, &bad_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (bad_global) {
            if (rank == 0) {
                fprintf(stderr, "Pivot too small at k=%zu, pivot=%g\n", k, pivot);
            }
            free(counts); free(displs);
            free(local); free(colbuf);
            return 0; // fail
        }

        /* --------- Faza L: licz L[i, k] dla i=k+1..n-1 (dzielimy po i) --------- */
        if (k + 1 < n) {
            size_t lenL = n - (k + 1);
            size_t sL, eL;
            block_range(lenL, p, rank, &sL, &eL);
            size_t locL = eL - sL;

            for (size_t t = 0; t < locL; t++) {
                size_t i = (k + 1) + (sL + t);
                long double sum = 0.0L;
                for (size_t m = 0; m < k; m++) {
                    sum += (long double)L[idx(n, i, m)] * (long double)U[idx(n, m, k)];
                }
                local[t] = (A[idx(n, i, k)] - (double)sum) / pivot;
            }

            for (int r = 0; r < p; r++) {
                size_t sr, er;
                block_range(lenL, p, r, &sr, &er);
                counts[r] = (int)(er - sr);
                displs[r] = (int)sr;
            }

            // zbierz fragmenty kolumny do bufora colbuf[0..lenL-1]
            MPI_Allgatherv(local, (int)locL, MPI_DOUBLE,
                           colbuf, counts, displs, MPI_DOUBLE,
                           MPI_COMM_WORLD);

            // wpisz do L[i,k]
            for (size_t t = 0; t < lenL; t++) {
                size_t i = (k + 1) + t;
                L[idx(n, i, k)] = colbuf[t];
            }
        }
        // (L[k,k] = 1 jest ustawione wcześniej; L[i,k]=0 dla i<k pozostaje 0)
    }

    free(counts); free(displs);
    free(local); free(colbuf);
    return 1; // ok
}

/* ---------- Wejście ---------- */
static void usage_rank0(void) {
    printf("Użycie:\n");
    printf("  mpirun -np <p> ./prir_lu_mpi <n> <seed> <repeats>\n");
    printf("Przykład:\n");
    printf("  mpirun -np 4 ./prir_lu_mpi 2000 123 3\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, p = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    long n_in = 0;
    unsigned long long seed_in = 0;
    long repeats_in = 0;

    if (rank == 0) {
        if (argc != 4) {
            usage_rank0();
            n_in = 0;
        } else {
            n_in = strtol(argv[1], NULL, 10);
            seed_in = strtoull(argv[2], NULL, 10);
            repeats_in = strtol(argv[3], NULL, 10);
        }
    }

    // broadcast parametrów
    MPI_Bcast(&n_in, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed_in, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&repeats_in, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (n_in <= 0 || repeats_in <= 0) {
        MPI_Finalize();
        return (n_in <= 0) ? 1 : 0;
    }

    size_t n = (size_t)n_in;
    int repeats = (int)repeats_in;
    uint64_t seed = (uint64_t)seed_in;

    // alokacje (pełne macierze na każdym ranku - prościej do ogarnięcia)
    size_t nn = n * n;
    double *A = (double *)malloc(nn * sizeof(double));
    double *L = (double *)malloc(nn * sizeof(double));
    double *U = (double *)malloc(nn * sizeof(double));
    if (!A || !L || !U) die_rank(rank, "malloc A/L/U failed (n too big?)");

    double t_sum = 0.0;
    double last_err = 0.0;

    for (int r = 0; r < repeats; r++) {
        // generacja A na rank 0, broadcast do wszystkich
        if (rank == 0) generate_A_diagonally_dominant(n, A, seed + (uint64_t)r);
        MPI_Bcast(A, (int)nn, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // init L, U
        memset(L, 0, nn * sizeof(double));
        memset(U, 0, nn * sizeof(double));
        for (size_t i = 0; i < n; i++) L[idx(n, i, i)] = 1.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        int ok = lu_doolittle_mpi(n, p, rank, A, L, U);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        // bierzemy czas najwolniejszego procesu (realny czas wykonania)
        double t_local = t1 - t0;
        double t_max = 0.0;
        MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        int ok_global = 0;
        MPI_Allreduce(&ok, &ok_global, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (!ok_global) {
            if (rank == 0) fprintf(stderr, "LU failed (pivot issue)\n");
            free(A); free(L); free(U);
            MPI_Finalize();
            return 2;
        }

        if (rank == 0) {
            t_sum += t_max;
        }

        // poprawność poza czasem (tylko rank 0)
        if (rank == 0) {
            double *M = (double *)calloc(nn, sizeof(double));
            if (!M) die_rank(rank, "calloc M failed");
            matmul_LU_minus_A(n, L, U, A, M);
            last_err = frob_norm(n, M);
            free(M);
        }
    }

    if (rank == 0) {
        double t_avg = t_sum / (double)repeats;
        printf("=== MPI LU Doolittle ===\n");
        printf("n=%zu p=%d repeats=%d seed=%llu\n", n, p, repeats, (unsigned long long)seed);
        printf("T_mpi(n,p)=%.6f s (avg, max over ranks)\n", t_avg);
        printf("||L*U-A||_F = %.6e\n", last_err);
    }

    free(A); free(L); free(U);
    MPI_Finalize();
    return 0;
}
