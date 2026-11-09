#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "matrix.h"

#define EPSILON 1e-7

/* Linear scan value retrieval helper for confirmation matching */
double get_coo_value(TSparseMatrix *M, unsigned int r, unsigned int c) {
    if (!M || !M->elements) return 0.0;
    
    for (unsigned int i = 0; i < M->nnz; i++) {
        if ((unsigned int)M->elements[i].r == r && (unsigned int)M->elements[i].c == c) {
            return M->elements[i].val;
        }
    }
    return 0.0;
}

/* Structural verification driver mapping thread calculation parity */
bool verify_results(TSparseMatrix *seq, TSparseMatrix *par) {
    if (!seq || !par) {
        fprintf(stderr, "Error: One or both matrices evaluate to NULL.\n");
        return false;
    }

    if (seq->nrows != par->nrows || seq->ncols != par->ncols) {
        fprintf(stderr, "Dimension Mismatch! Sequential: %ux%u, Parallel: %ux%u\n", 
                seq->nrows, seq->ncols, par->nrows, par->ncols);
        return false;
    }

    printf("Verifying matrix elements against ground truth (Tolerance = %e)...\n", EPSILON);

    for (unsigned int i = 0; i < seq->nnz; i++) {
        unsigned int r = seq->elements[i].r;
        unsigned int c = seq->elements[i].c;
        double val_seq = seq->elements[i].val;
        double val_par = get_coo_value(par, r, c);

        if (fabs(val_seq - val_par) > EPSILON) {
            fprintf(stderr, "Mismatch detected at index [%u][%u]! Seq value: %f, Par value: %f\n", 
                    r, c, val_seq, val_par);
            return false;
        }
    }

    for (unsigned int i = 0; i < par->nnz; i++) {
        unsigned int r = par->elements[i].r;
        unsigned int c = par->elements[i].c;
        double val_par = par->elements[i].val;
        double val_seq = get_coo_value(seq, r, c);

        if (fabs(val_seq - val_par) > EPSILON) {
            fprintf(stderr, "Phantom mismatch detected at index [%u][%u]! Seq value: %f, Par value: %f\n", 
                    r, c, val_seq, val_par);
            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <MatrixA.mtx> <MatrixB.mtx> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *fileA = argv[1];
    const char *fileB = argv[2];
    int num_threads = atoi(argv[3]);

    if (num_threads <= 0) {
        fprintf(stderr, "Error: Thread allocation must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    printf("Loading sparse datasets into memory structures...\n");
    TSparseMatrix *A = read_coo_matrix(fileA);
    TSparseMatrix *B = read_coo_matrix(fileB);

    if (!A || !B) {
        fprintf(stderr, "Error loading input matrices.\n");
        free_sparse_matrix(A);
        free_sparse_matrix(B);
        return EXIT_FAILURE;
    }

    printf("Matrix A Loaded: %u x %u (%u NNZ)\n", A->nrows, A->ncols, A->nnz);
    printf("Matrix B Loaded: %u x %u (%u NNZ)\n", B->nrows, B->ncols, B->nnz);

    printf("Executing single-threaded baseline matrix calculation...\n");
    TSparseMatrix *res_seq = spmm_sequential(A, B);

    printf("Executing parallel multi-core engine with %d threads...\n", num_threads);
    TSparseMatrix *res_par = spmm_parallel(A, B, num_threads);

    bool success = verify_results(res_seq, res_par);

    free_sparse_matrix(A);
    free_sparse_matrix(B);
    free_sparse_matrix(res_seq);
    free_sparse_matrix(res_par);

    if (success) {
        printf("\n>> SUCCESS: Multithreaded execution calculations mirror structural ground truth flawlessly!\n");
        return EXIT_SUCCESS;
    } else {
        printf("\n>> FAILURE: Discrepancy errors discovered between thread processing executions.\n");
        return EXIT_FAILURE;
    }
}
