/*
PRiR Projekt, Etap I (TYLKO) - Zadanie 11
Faktoryzacja LU metodą Doolittle’a: A = L * U, diag(L)=1
Wariant równoległy: procesy współbieżne (fork) + pamięć współdzielona (shm_open/mmap)
+ semafory POSIX (sem_init) + bariery.

Założenia:
- C, styl proceduralny, bez obiektów.
- Tylko wersja równoległa (p procesów).
- Testy dla różnych n i p; repeats dla uśrednienia.
- Sprawdzenie poprawności: ||L*U - A||_F (norma Frobeniusa), NIE wliczane do czasu.
- Doolittle bez pivotowania: generujemy A jako diagonally dominant, żeby pivoty były stabilne.

Kompilacja (Linux):
  gcc -std=c11 -O2 -Wall -Wextra -pedantic -pthread prir_lu_proc_only.c -lm -o prir_lu_proc

Uruchomienie (interaktywnie):
  ./prir_lu_proc

Uruchomienie (opcjonalnie z parametrami):
  ./prir_lu_proc n p seed repeats
*/

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* ----------------------------- Utils ----------------------------- */

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

static void die_msg(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static int64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        die("clock_gettime");
    }
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static double elapsed_s(int64_t t0, int64_t t1) {
    return (double)(t1 - t0) / 1e9;
}

static void print_time_s(const char *label, double seconds) {
    printf("[TIME] %s: %.3f s\n", label, seconds);
}

static size_t idx(size_t n, size_t i, size_t j) {
    return i * n + j;
}

/* Prosty RNG: xorshift64* */
static uint64_t xorshift64s(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

/* Losuj double z [-1, 1] */
static double rnd_u11(uint64_t *state) {
    uint64_t r = xorshift64s(state);
    uint64_t mant = r >> 11; /* 53 bity */
    double u = (double)mant / (double)((1ULL << 53) - 1ULL); /* [0,1] */
    return 2.0 * u - 1.0;
}

/* --------------------------- Frobenius --------------------------- */

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
    /* Out = L*U - A */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            long double sum = 0.0L;
            size_t k0 = (j < i) ? j : i;
            for (size_t k = 0; k <= k0; k++) {
                sum += (long double)L[idx(n, i, k)] * (long double)U[idx(n, k, j)];
            }
            Out[idx(n, i, j)] = (double)sum - A[idx(n, i, j)];
        }
    }
}

/* ------------------------ Współdzielona pamięć ------------------------ */

typedef struct {
    size_t n;
    int p;

    sem_t b1_mutex;
    sem_t b1_turnstile;
    sem_t b2_mutex;
    sem_t b2_turnstile;
    int b1_count;
    int b2_count;

    sem_t err_mutex;
    int error_flag;
    size_t error_k;
    double pivot_value;
} SharedCtrl;

typedef struct {
    SharedCtrl ctrl;
    /* dalej w pamięci: A, L, U jako double[n*n] */
} SharedBlock;

static double *shm_A(SharedBlock *blk) {
    return (double *)((char *)blk + sizeof(SharedBlock));
}
static double *shm_L(SharedBlock *blk, size_t n) {
    return shm_A(blk) + (n * n);
}
static double *shm_U(SharedBlock *blk, size_t n) {
    return shm_L(blk, n) + (n * n);
}

/* ---------------------------- Bariery ---------------------------- */

static void barrier_wait(sem_t *mutex, sem_t *turnstile, int *count, int p) {
    if (sem_wait(mutex) != 0) die("sem_wait(mutex)");
    (*count)++;
    if (*count == p) {
        for (int i = 0; i < p; i++) {
            if (sem_post(turnstile) != 0) die("sem_post(turnstile)");
        }
        *count = 0;
    }
    if (sem_post(mutex) != 0) die("sem_post(mutex)");
    if (sem_wait(turnstile) != 0) die("sem_wait(turnstile)");
}

static void set_error(SharedCtrl *c, size_t k, double pivot) {
    if (sem_wait(&c->err_mutex) != 0) die("sem_wait(err_mutex)");
    if (!c->error_flag) {
        c->error_flag = 1;
        c->error_k = k;
        c->pivot_value = pivot;
    }
    if (sem_post(&c->err_mutex) != 0) die("sem_post(err_mutex)");
}

/* ----------------------- Doolittle (procesy) ----------------------- */

static void child_work(SharedBlock *blk, int rank) {
    SharedCtrl *c = &blk->ctrl;
    size_t n = c->n;
    int p = c->p;

    double *A = shm_A(blk);
    double *L = shm_L(blk, n);
    double *U = shm_U(blk, n);

    for (size_t k = 0; k < n; k++) {
        if (c->error_flag) break;

        /* Faza U: rozdziel j = k..n-1 */
        size_t lenU = n - k;
        size_t start = (size_t)rank * lenU / (size_t)p;
        size_t end = (size_t)(rank + 1) * lenU / (size_t)p;

        for (size_t t = start; t < end; t++) {
            size_t j = k + t;
            long double sum = 0.0L;
            for (size_t m = 0; m < k; m++) {
                sum += (long double)L[idx(n, k, m)] * (long double)U[idx(n, m, j)];
            }
            U[idx(n, k, j)] = A[idx(n, k, j)] - (double)sum;
        }

        barrier_wait(&c->b1_mutex, &c->b1_turnstile, &c->b1_count, p);

        double pivot = U[idx(n, k, k)];
        if (fabs(pivot) < 1e-12) {
            set_error(c, k, pivot);
        }

        /* Faza L: rozdziel i = k+1..n-1 */
        if (k + 1 < n) {
            size_t lenL = n - (k + 1);
            size_t startL = (size_t)rank * lenL / (size_t)p;
            size_t endL = (size_t)(rank + 1) * lenL / (size_t)p;

            for (size_t t = startL; t < endL; t++) {
                size_t i = (k + 1) + t;
                long double sum = 0.0L;
                for (size_t m = 0; m < k; m++) {
                    sum += (long double)L[idx(n, i, m)] * (long double)U[idx(n, m, k)];
                }
                L[idx(n, i, k)] = (A[idx(n, i, k)] - (double)sum) / pivot;
            }
        }

        barrier_wait(&c->b2_mutex, &c->b2_turnstile, &c->b2_count, p);
    }

    _exit(0);
}

/* -------------------------- Generowanie A -------------------------- */

static void generate_A_diagonally_dominant(size_t n, double *A, uint64_t seed) {
    uint64_t st = seed ? seed : 1ULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A[idx(n, i, j)] = rnd_u11(&st);
        }
    }

    for (size_t i = 0; i < n; i++) {
        long double sum = 0.0L;
        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;
            sum += fabsl((long double)A[idx(n, i, j)]);
        }
        A[idx(n, i, i)] += (double)(sum + 1.0L);
    }
}

/* ------------------------- SHM init/cleanup ------------------------- */

typedef struct {
    char name[64];
    int fd;
    size_t bytes;
    SharedBlock *blk;
} ShmHandle;

static void shm_cleanup(ShmHandle *h) {
    if (!h) return;
    if (h->blk && h->blk != MAP_FAILED) {
        munmap(h->blk, h->bytes);
        h->blk = NULL;
    }
    if (h->fd >= 0) {
        close(h->fd);
        h->fd = -1;
    }
    if (h->name[0]) {
        shm_unlink(h->name);
        h->name[0] = '\0';
    }
}

static ShmHandle shm_create(size_t n) {
    ShmHandle h;
    memset(&h, 0, sizeof(h));
    h.fd = -1;

    pid_t pid = getpid();
    snprintf(h.name, sizeof(h.name), "/prir_lu_%ld", (long)pid);

    size_t nn = n * n;
    h.bytes = sizeof(SharedBlock) + 3 * nn * sizeof(double);

    h.fd = shm_open(h.name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (h.fd < 0) die("shm_open");

    if (ftruncate(h.fd, (off_t)h.bytes) != 0) {
        shm_cleanup(&h);
        die("ftruncate");
    }

    void *ptr = mmap(NULL, h.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, h.fd, 0);
    if (ptr == MAP_FAILED) {
        shm_cleanup(&h);
        die("mmap");
    }

    h.blk = (SharedBlock *)ptr;
    memset(h.blk, 0, h.bytes);
    return h;
}

static void sem_destroy_safe(sem_t *s) {
    if (sem_destroy(s) != 0) {
        perror("sem_destroy");
    }
}

/* ------------------------------ CLI ------------------------------ */

static int read_line(char *buf, size_t buflen) {
    if (!fgets(buf, (int)buflen, stdin)) return 0;
    size_t len = strlen(buf);
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) {
        buf[--len] = '\0';
    }
    return 1;
}

static long read_long_prompt(const char *prompt, long min_value) {
    char line[128];
    for (;;) {
        printf("%s", prompt);
        fflush(stdout);
        if (!read_line(line, sizeof(line))) die_msg("Brak danych wejściowych");
        errno = 0;
        char *end = NULL;
        long v = strtol(line, &end, 10);
        if (errno == 0 && end && *end == '\0' && v >= min_value) return v;
        printf("Błędna wartość. Spróbuj ponownie.\n");
    }
}

static unsigned long long read_ull_prompt(const char *prompt) {
    char line[128];
    for (;;) {
        printf("%s", prompt);
        fflush(stdout);
        if (!read_line(line, sizeof(line))) die_msg("Brak danych wejściowych");
        errno = 0;
        char *end = NULL;
        unsigned long long v = strtoull(line, &end, 10);
        if (errno == 0 && end && *end == '\0') return v;
        printf("Błędna wartość. Spróbuj ponownie.\n");
    }
}

static void ask_return_to_main(void) {
    char line[16];
    printf("\nCzy chcesz wrócić do pliku głównego (./main)? [t/n]: ");
    fflush(stdout);

    if (!fgets(line, sizeof(line), stdin)) return;

    if (line[0] == 't' || line[0] == 'T') {
        printf("[INFO] Uruchamiam ./main\n\n");
        execl("./main", "./main", (char *)NULL);
        perror("execl ./main");
    }
}

static void print_matrix(const char *name, size_t n, const double *M) {
    printf("\n%s =\n", name);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            printf("%10.4f ", M[idx(n, i, j)]);
        }
        printf("\n");
    }
}

/* ------------------------------ Main ------------------------------ */

int main(int argc, char **argv) {
    int64_t t_program_start = now_ns();

    size_t n = 0;
    int p = 0;
    uint64_t seed = 0;
    int repeats = 0;

    if (argc == 5) {
        char *end = NULL;

        errno = 0;
        long n_long = strtol(argv[1], &end, 10);
        if (errno || !end || *end || n_long <= 0) die_msg("Błędne n");
        n = (size_t)n_long;

        errno = 0;
        long p_long = strtol(argv[2], &end, 10);
        if (errno || !end || *end || p_long <= 0) die_msg("Błędne p");
        p = (int)p_long;

        errno = 0;
        unsigned long long seed_ull = strtoull(argv[3], &end, 10);
        if (errno || !end || *end) die_msg("Błędne seed");
        seed = (uint64_t)seed_ull;

        errno = 0;
        long rep_long = strtol(argv[4], &end, 10);
        if (errno || !end || *end || rep_long <= 0) die_msg("Błędne repeats");
        repeats = (int)rep_long;
    } else {
        printf("PRiR Zadanie 11, LU Doolittle, procesy współbieżne\n");
        printf("Tryb interaktywny\n\n");

        long n_long = read_long_prompt("n (rozmiar macierzy, >=1): ", 1);
        long p_long = read_long_prompt("p (liczba procesów, >=1): ", 1);
        unsigned long long seed_ull = read_ull_prompt("seed (0..): ");
        long rep_long = read_long_prompt("repeats (liczba powtórzeń, >=1): ", 1);

        n = (size_t)n_long;
        p = (int)p_long;
        seed = (uint64_t)seed_ull;
        repeats = (int)rep_long;
    }

    printf("\n[INFO] Parametry: n=%zu, p=%d, repeats=%d, seed=%llu\n",
           n, p, repeats, (unsigned long long)seed);

    double t_proc_s_avg = 0.0;
    double err_proc_last = 0.0;

    printf("\n[INFO] Obliczenia równoległe, procesy współbieżne...\n");

    for (int r = 0; r < repeats; r++) {
        printf("\n[INFO] Powtórzenie %d/%d: inicjalizacja pamięci współdzielonej i semaforów...\n",
               r + 1, repeats);

        int64_t t_init0 = now_ns();

        ShmHandle shm = shm_create(n);
        SharedBlock *blk = shm.blk;
        SharedCtrl *c = &blk->ctrl;

        c->n = n;
        c->p = p;
        c->b1_count = 0;
        c->b2_count = 0;
        c->error_flag = 0;
        c->error_k = 0;
        c->pivot_value = 0.0;

        if (sem_init(&c->b1_mutex, 1, 1) != 0) { shm_cleanup(&shm); die("sem_init b1_mutex"); }
        if (sem_init(&c->b1_turnstile, 1, 0) != 0) { shm_cleanup(&shm); die("sem_init b1_turnstile"); }
        if (sem_init(&c->b2_mutex, 1, 1) != 0) { shm_cleanup(&shm); die("sem_init b2_mutex"); }
        if (sem_init(&c->b2_turnstile, 1, 0) != 0) { shm_cleanup(&shm); die("sem_init b2_turnstile"); }
        if (sem_init(&c->err_mutex, 1, 1) != 0) { shm_cleanup(&shm); die("sem_init err_mutex"); }

        double *A = shm_A(blk);
        double *L = shm_L(blk, n);
        double *U = shm_U(blk, n);

        generate_A_diagonally_dominant(n, A, seed + (uint64_t)r);
        memset(L, 0, n * n * sizeof(double));
        memset(U, 0, n * n * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            L[idx(n, i, i)] = 1.0;
        }

        int64_t t_init1 = now_ns();
        print_time_s("Po inicjalizacji (procesy)", elapsed_s(t_init0, t_init1));

        pid_t *pids = (pid_t *)calloc((size_t)p, sizeof(pid_t));
        if (!pids) {
            shm_cleanup(&shm);
            die("calloc pids");
        }

        printf("[INFO] Start obliczeń LU w procesach (fork)...\n");
        int64_t t0 = now_ns();

        for (int rank = 0; rank < p; rank++) {
            pid_t pid = fork();
            if (pid < 0) {
                for (int k = 0; k < rank; k++) {
                    if (pids[k] > 0) kill(pids[k], SIGKILL);
                }
                free(pids);
                shm_cleanup(&shm);
                die("fork");
            }
            if (pid == 0) {
                child_work(blk, rank);
            }
            pids[rank] = pid;
        }

        int status = 0;
        for (int i = 0; i < p; i++) {
            if (waitpid(pids[i], &status, 0) < 0) {
                free(pids);
                shm_cleanup(&shm);
                die("waitpid");
            }
        }

        int64_t t1 = now_ns();
        double t_proc_s = elapsed_s(t0, t1);
        print_time_s("Po zakończeniu obliczeń (procesy)", t_proc_s);

        free(pids);

        if (c->error_flag) {
            fprintf(stderr, "Procesy: pivot za mały w k=%zu, pivot=%g\n", c->error_k, c->pivot_value);
            sem_destroy_safe(&c->err_mutex);
            sem_destroy_safe(&c->b2_turnstile);
            sem_destroy_safe(&c->b2_mutex);
            sem_destroy_safe(&c->b1_turnstile);
            sem_destroy_safe(&c->b1_mutex);
            shm_cleanup(&shm);

            ask_return_to_main();

            return EXIT_FAILURE;
        }

        size_t nn = n * n;
        double *M = (double *)calloc(nn, sizeof(double));
        if (!M) {
            shm_cleanup(&shm);
            die("calloc M");
        }

        printf("[INFO] Sprawdzenie poprawności (poza czasem)...\n");
        matmul_LU_minus_A(n, L, U, A, M);
        err_proc_last = frob_norm(n, M);

        if (n <= 10) {
            print_matrix("L", n, L);
            print_matrix("U", n, U);
        }

        free(M);

        sem_destroy_safe(&c->err_mutex);
        sem_destroy_safe(&c->b2_turnstile);
        sem_destroy_safe(&c->b2_mutex);
        sem_destroy_safe(&c->b1_turnstile);
        sem_destroy_safe(&c->b1_mutex);
        shm_cleanup(&shm);

        t_proc_s_avg += t_proc_s;
    }

    t_proc_s_avg /= (double)repeats;

    printf("\n=== PODSUMOWANIE ===\n");
    printf("n=%zu p=%d repeats=%d seed=%llu\n", n, p, repeats, (unsigned long long)seed);
    printf("T_proc(n,p)=%.6f s (avg)\n", t_proc_s_avg);
    printf("||L*U-A||_F proc=%.6e\n", err_proc_last);

    int64_t t_program_end = now_ns();
    print_time_s("Calkowity czas dzialania programu", elapsed_s(t_program_start, t_program_end));

    ask_return_to_main();

    return EXIT_SUCCESS;
}
