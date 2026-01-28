#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <math.h>




static int write_all(int fd, const void *buf, size_t nbytes) {
    const char *p = (const char*)buf;
    size_t left = nbytes;
    while (left > 0) {
        ssize_t w = write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += (size_t)w;
        left -= (size_t)w;
    }
    return 0;
}

static int read_all(int fd, void *buf, size_t nbytes) {
    char *p = (char*)buf;
    size_t left = nbytes;
    while (left > 0) {
        ssize_t r = read(fd, p, left);
        if (r == 0) return -1;         // EOF
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += (size_t)r;
        left -= (size_t)r;
    }
    return 0;
}

typedef struct {
    int fd;
    int start;
    int len;
    pid_t pid;
} job_t;

/* ======== LU (DOOLITTLE) – “komunikaty” przez pipe ======== */
void komunikaty(double *A, double *L, double *U, int n)
{
    int p = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (p < 1) p = 1;
    if (p > n) p = n;   // więcej procesów niż elementów nie ma sensu

    job_t *jobs = (job_t*)malloc((size_t)p * sizeof(job_t));
    if (!jobs) { perror("malloc"); return; }

    for (int k = 0; k < n; k++) {

        /* ===================== 1) LICZ U[k][j] (j=k..n-1) ===================== */
        int totalU = n - k;
        int chunkU = (totalU + p - 1) / p;
        int job_count = 0;

        for (int w = 0; w < p; w++) {
            int j0 = k + w * chunkU;
            int j1 = j0 + chunkU;
            if (j1 > n) j1 = n;
            if (j0 >= n) continue;

            int fds[2];
            if (pipe(fds) != 0) { perror("pipe"); goto cleanup; }

            pid_t pid = fork();
            if (pid < 0) { perror("fork"); close(fds[0]); close(fds[1]); goto cleanup; }

            if (pid == 0) {
                // dziecko: liczy swój fragment i wysyła wyniki
                close(fds[0]);

                int len = j1 - j0;
                double *out = (double*)malloc((size_t)len * sizeof(double));
                if (!out) _exit(1);

                for (int j = j0; j < j1; j++) {
                    double sum = 0.0;
                    for (int m = 0; m < k; m++) {
                        sum += L[k*n + m] * U[m*n + j];
                    }
                    out[j - j0] = A[k*n + j] - sum;
                }

                if (write_all(fds[1], out, (size_t)len * sizeof(double)) != 0) {
                    free(out);
                    _exit(2);
                }

                free(out);
                close(fds[1]);
                _exit(0);
            }

            // rodzic
            close(fds[1]);
            jobs[job_count++] = (job_t){ .fd = fds[0], .start = j0, .len = (j1 - j0), .pid = pid };
        }

        // odbiór wyników U
        for (int t = 0; t < job_count; t++) {
            double *buf = (double*)malloc((size_t)jobs[t].len * sizeof(double));
            if (!buf) { perror("malloc"); goto cleanup; }

            if (read_all(jobs[t].fd, buf, (size_t)jobs[t].len * sizeof(double)) != 0) {
                perror("read");
                free(buf);
                goto cleanup;
            }

            memcpy(&U[k*n + jobs[t].start], buf, (size_t)jobs[t].len * sizeof(double));
            free(buf);
            close(jobs[t].fd);
        }
        for (int t = 0; t < job_count; t++) {
            waitpid(jobs[t].pid, NULL, 0);
        }

        // zabezpieczenie: dzielenie przez U[k][k]
        if (fabs(U[k*n + k]) < 1e-15) {
            fprintf(stderr, "Blad numeryczny: U[%d][%d] ~ 0 (brak pivotingu)\n", k, k);
            goto cleanup;
        }

        /* ===================== 2) LICZ L[i][k] (i=k+1..n-1) ===================== */
        int totalL = n - (k + 1);
        if (totalL <= 0) continue;

        int chunkL = (totalL + p - 1) / p;
        job_count = 0;

        for (int w = 0; w < p; w++) {
            int i0 = (k + 1) + w * chunkL;
            int i1 = i0 + chunkL;
            if (i1 > n) i1 = n;
            if (i0 >= n) continue;

            int fds[2];
            if (pipe(fds) != 0) { perror("pipe"); goto cleanup; }

            pid_t pid = fork();
            if (pid < 0) { perror("fork"); close(fds[0]); close(fds[1]); goto cleanup; }

            if (pid == 0) {
                close(fds[0]);

                int len = i1 - i0;
                double *out = (double*)malloc((size_t)len * sizeof(double));
                if (!out) _exit(3);

                for (int i = i0; i < i1; i++) {
                    double sum = 0.0;
                    for (int m = 0; m < k; m++) {
                        sum += L[i*n + m] * U[m*n + k];
                    }
                    out[i - i0] = (A[i*n + k] - sum) / U[k*n + k];
                }

                if (write_all(fds[1], out, (size_t)len * sizeof(double)) != 0) {
                    free(out);
                    _exit(4);
                }

                free(out);
                close(fds[1]);
                _exit(0);
            }

            close(fds[1]);
            jobs[job_count++] = (job_t){ .fd = fds[0], .start = i0, .len = (i1 - i0), .pid = pid };
        }

        // odbiór wyników L
        for (int t = 0; t < job_count; t++) {
            double *buf = (double*)malloc((size_t)jobs[t].len * sizeof(double));
            if (!buf) { perror("malloc"); goto cleanup; }

            if (read_all(jobs[t].fd, buf, (size_t)jobs[t].len * sizeof(double)) != 0) {
                perror("read");
                free(buf);
                goto cleanup;
            }

            for (int idx = 0; idx < jobs[t].len; idx++) {
                int i = jobs[t].start + idx;
                L[i*n + k] = buf[idx];
            }

            free(buf);
            close(jobs[t].fd);
        }
        for (int t = 0; t < job_count; t++) {
            waitpid(jobs[t].pid, NULL, 0);
        }
    }

cleanup:
    free(jobs);
}
