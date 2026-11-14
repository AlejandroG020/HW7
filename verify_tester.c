#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h> // Added for timing

#include "matrix.h"

#define EPSILON 1e-7

/* Required thread configurations */
#define NUM_CONFIGS 5

static const int thread_counts[NUM_CONFIGS] = {
    1, 2, 4, 8, 16
};

/* Stopwatch function */
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Compare COO elements by row, then column */
static int compare_coo(const void *a, const void *b) {
    const COOElement *x = (const COOElement *)a;
    const COOElement *y = (const COOElement *)b;

    if (x->r != y->r)
        return (x->r > y->r) - (x->r < y->r);

    return (x->c > y->c) - (x->c < y->c);
}

/* Verify sequential and parallel results */
bool verify_results(
    TSparseMatrix *seq,
    TSparseMatrix *par
) {
    if (!seq || !par) {
        fprintf(stderr, "ERROR: A result matrix is NULL.\n");
        return false;
    }

    /* Check matrix dimensions */
    if (seq->nrows != par->nrows ||
        seq->ncols != par->ncols) {

        fprintf(stderr, "ERROR: Dimension mismatch.\n");
        return false;
    }

    /* Both results should have the same number of entries */
    if (seq->nnz != par->nnz) {
        fprintf(stderr,
                "ERROR: NNZ mismatch. Sequential: %u, Parallel: %u\n",
                seq->nnz, par->nnz);
        return false;
    }

    /* Make sure the element arrays exist when needed */
    if (seq->nnz > 0 &&
        (!seq->elements || !par->elements)) {

        fprintf(stderr, "ERROR: Missing COO element array.\n");
        return false;
    }

    unsigned int count = seq->nnz;

    /* Create copies so original matrices remain unchanged */
    COOElement *seq_copy = NULL;
    COOElement *par_copy = NULL;

    if (count > 0) {
        seq_copy = malloc(count * sizeof(COOElement));
        par_copy = malloc(count * sizeof(COOElement));

        if (!seq_copy || !par_copy) {
            fprintf(stderr, "ERROR: Memory allocation failed.\n");
            free(seq_copy);
            free(par_copy);
            return false;
        }

        for (unsigned int i = 0; i < count; i++) {
            seq_copy[i] = seq->elements[i];
            par_copy[i] = par->elements[i];
        }

        /* Sort both matrices into the same coordinate order */
        qsort(seq_copy, count, sizeof(COOElement), compare_coo);
        qsort(par_copy, count, sizeof(COOElement), compare_coo);
    }

    printf("Checking %u nonzero entries...\n", count);

    bool success = true;

    for (unsigned int i = 0; i < count; i++) {

        COOElement s = seq_copy[i];
        COOElement p = par_copy[i];

        /* Check valid coordinates */
        if (s.r < 0 || p.r < 0 ||
            s.c < 0 || p.c < 0 ||
            (unsigned int)s.r >= seq->nrows ||
            (unsigned int)p.r >= par->nrows ||
            (unsigned int)s.c >= seq->ncols ||
            (unsigned int)p.c >= par->ncols) {

            fprintf(stderr,
                    "ERROR: Invalid coordinate at entry %u.\n", i);

            success = false;
            break;
        }

        /* Detect duplicate coordinates */
        if (i > 0 &&
            (s.r == seq_copy[i - 1].r &&
             s.c == seq_copy[i - 1].c)) {

            fprintf(stderr,
                    "ERROR: Duplicate sequential coordinate [%d][%d].\n",
                    s.r, s.c);

            success = false;
            break;
        }

        if (i > 0 &&
            (p.r == par_copy[i - 1].r &&
             p.c == par_copy[i - 1].c)) {

            fprintf(stderr,
                    "ERROR: Duplicate parallel coordinate [%d][%d].\n",
                    p.r, p.c);

            success = false;
            break;
        }

        /* Check that coordinates match */
        if (s.r != p.r || s.c != p.c) {
            fprintf(stderr,
                    "ERROR: Coordinate mismatch.\n"
                    "Sequential: [%d][%d]\n"
                    "Parallel:   [%d][%d]\n",
                    s.r, s.c, p.r, p.c);

            success = false;
            break;
        }

        /* Check floating-point values */
        if (!isfinite(s.val) || !isfinite(p.val) ||
            fabs(s.val - p.val) > EPSILON) {

            fprintf(stderr,
                    "ERROR: Value mismatch at [%d][%d].\n"
                    "Sequential: %.12f\n"
                    "Parallel:   %.12f\n",
                    s.r, s.c, s.val, p.val);

            success = false;
            break;
        }
    }

    free(seq_copy);
    free(par_copy);

    return success;
}

/* Main verification and concurrency stress testing */
int main(int argc, char *argv[]) {

    if (argc != 3) {
        fprintf(stderr,
                "Usage: %s <MatrixA.mtx> <MatrixB.mtx>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *fileA = argv[1];
    const char *fileB = argv[2];

    printf("========================================\n");
    printf("       SPARSE MATRIX VERIFICATION\n");
    printf("========================================\n");

    /* Load input matrices */
    printf("\nLoading input matrices...\n");

    TSparseMatrix *A = read_coo_matrix(fileA);
    TSparseMatrix *B = read_coo_matrix(fileB);

    if (!A || !B) {
        fprintf(stderr, "ERROR: Failed to load matrices.\n");

        free_sparse_matrix(A);
        free_sparse_matrix(B);

        return EXIT_FAILURE;
    }

    printf("Matrix A: %u x %u | NNZ: %u\n",
           A->nrows, A->ncols, A->nnz);

    printf("Matrix B: %u x %u | NNZ: %u\n",
           B->nrows, B->ncols, B->nnz);

    /* Check multiplication compatibility */
    if (A->ncols != B->nrows) {
        fprintf(stderr,
                "ERROR: Matrix dimensions are incompatible.\n");

        free_sparse_matrix(A);
        free_sparse_matrix(B);

        return EXIT_FAILURE;
    }

    /* Calculate sequential ground truth */
    printf("\nRunning sequential baseline...\n");

    double start = now_seconds();
    TSparseMatrix *res_seq = spmm_sequential(A, B);
    double seq_time = now_seconds() - start;

    if (!res_seq) {
        fprintf(stderr,
                "ERROR: Sequential multiplication failed.\n");

        free_sparse_matrix(A);
        free_sparse_matrix(B);

        return EXIT_FAILURE;
    }

    printf("Sequential Time: %.6f seconds\n", seq_time);

    bool all_passed = true;

    /* Test all required thread configurations */
    printf("\n========================================\n");
    printf("       CONCURRENCY STRESS TEST\n");
    printf("========================================\n");
    printf("%8s %15s %12s %10s\n", "Threads", "Time (seconds)", "Speedup", "Status");
    printf("----------------------------------------------------\n");

    for (int i = 0; i < NUM_CONFIGS; i++) {

        int threads = thread_counts[i];

        start = now_seconds();
        TSparseMatrix *res_par =
            spmm_parallel(A, B, threads);
        double par_time = now_seconds() - start;

        if (!res_par) {
            fprintf(stderr,
                    "FAILURE: Parallel calculation failed "
                    "with %d threads.\n",
                    threads);

            all_passed = false;
            continue;
        }

        bool passed = verify_results(res_seq, res_par);
        double speedup = seq_time / par_time;

        if (passed) {
            printf("%8d %15.6f %12.3fx %10s\n", threads, par_time, speedup, "PASS");
        } else {
            printf("%8d %15.6f %12.3fx %10s\n", threads, par_time, speedup, "FAIL");

            all_passed = false;
        }

        free_sparse_matrix(res_par);
    }

    /* Final summary */
    printf("\n========================================\n");

    if (all_passed) {
        printf("SUCCESS: ALL THREAD CONFIGURATIONS PASSED!\n");
    } else {
        printf("FAILURE: ONE OR MORE TESTS FAILED.\n");
    }

    printf("========================================\n");

    /* Release all allocated memory */
    free_sparse_matrix(res_seq);
    free_sparse_matrix(A);
    free_sparse_matrix(B);

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}