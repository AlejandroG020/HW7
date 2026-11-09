#ifndef __SPARSE_MATRIX_H__
#define __SPARSE_MATRIX_H__

typedef struct {
    int r;          // Row index
    int c;          // Column index
    double val;     // Matrix entry value
} COOElement;

typedef struct Matrix {
    unsigned int nrows;
    unsigned int ncols;
    unsigned int nnz;      // Total number of non-zero entries
    COOElement *elements;  // Array of non-zero elements
} TSparseMatrix;

// Core APIs to be implemented and verified by your group
TSparseMatrix *read_coo_matrix(const char *filename);
void free_sparse_matrix(TSparseMatrix *m);
TSparseMatrix *spmm_sequential(TSparseMatrix *A, TSparseMatrix *B);
TSparseMatrix *spmm_parallel(TSparseMatrix *A, TSparseMatrix *B, int num_threads);

#endif