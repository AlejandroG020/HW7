#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include "matrix.h"
#include "mmio.h"

/* Ingestion engine utilizing the official NIST Matrix Market Library */
TSparseMatrix *read_coo_matrix(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Failed to open matrix file");
        return NULL;
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        fclose(f);
        return NULL;
    }

    // Validation rule: verify format is real, coordinate matrix
    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode) || !mm_is_real(matcode)) {
        fprintf(stderr, "Unsupported matrix type. Must be a real, coordinate matrix.\n");
        fclose(f);
        return NULL;
    }

    // We can handle "general", "symmetric" and "skew-symmetric" files
    if (!mm_is_general(matcode) && !mm_is_symmetric(matcode) && !mm_is_skew(matcode)) {
        fprintf(stderr, "Unsupported matrix type. Must be general, symmetric or skew-symmetric.\n");
        fclose(f);
        return NULL;
    }

    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Failed to parse matrix dimension size tokens.\n");
        fclose(f);
        return NULL;
    }

    // Symmetric files only list half of the matrix (one triangle), so for every
    // entry that is not on the diagonal we also have to add its mirror image.
    // That means we may need up to 2*nz slots.
    int mirror = !mm_is_general(matcode);
    size_t capacity = mirror ? 2 * (size_t)nz : (size_t)nz;
    if (capacity == 0) capacity = 1;

    TSparseMatrix *matrix = malloc(sizeof(TSparseMatrix));
    if (!matrix) {
        fclose(f);
        return NULL;
    }

    matrix->nrows = M;
    matrix->ncols = N;
    matrix->nnz = 0;
    matrix->elements = malloc(capacity * sizeof(COOElement));
    if (!matrix->elements) {
        free(matrix);
        fclose(f);
        return NULL;
    }

    int count = 0;     // how many entries we have stored so far
    for (int i = 0; i < nz; i++) {
        int row, col;
        double value;
        // Matrix Market format is 1-indexed; convert down to standard 0-indexed structures
        if (fscanf(f, "%d %d %lg\n", &row, &col, &value) != 3) {
            fprintf(stderr, "Error parsing element line %d\n", i);
            free(matrix->elements);
            free(matrix);
            fclose(f);
            return NULL;
        }
        // make sure the file isn't pointing outside the matrix
        if (row < 1 || row > M || col < 1 || col > N) {
            fprintf(stderr, "Element line %d is outside the matrix.\n", i);
            free(matrix->elements);
            free(matrix);
            fclose(f);
            return NULL;
        }
        matrix->elements[count].r = row - 1;
        matrix->elements[count].c = col - 1;
        matrix->elements[count].val = value;
        count++;

        // add the mirror image (skip the diagonal, it's its own mirror)
        if (mirror && row != col) {
            matrix->elements[count].r = col - 1;
            matrix->elements[count].c = row - 1;
            matrix->elements[count].val = mm_is_skew(matcode) ? -value : value;
            count++;
        }
    }
    matrix->nnz = count;

    fclose(f);
    return matrix;
}

void free_sparse_matrix(TSparseMatrix *m) {
    if (m) {
        if (m->elements) free(m->elements);
        free(m);
    }
}

/* ======================================================================
 * Stuff shared by the sequential AND the parallel version.
 *
 * The plan (it's called Gustavson's method, in case a TA asks):
 *   1. Turn A and B into "CSR" form, so we can quickly find all the
 *      entries of any one row.
 *   2. Do the multiplication one row of A at a time. Row i of the answer
 *      only depends on row i of A (and all of B), so rows don't affect
 *      each other. That's what makes it easy to share between threads.
 * ====================================================================== */

/* CSR = "compressed sparse row". It's the same entries as the COO list,
 * but grouped by row, plus a little table (rowptr) saying where each row
 * starts. Row i lives in positions rowptr[i] up to rowptr[i+1]-1. */
typedef struct {
    unsigned int nrows;
    unsigned int ncols;
    int *rowptr;      // nrows+1 numbers
    int *col;         // column of each stored entry
    double *val;      // value of each stored entry
} CSRMatrix;

static void free_csr(CSRMatrix *csr) {
    if (csr) {
        free(csr->rowptr);
        free(csr->col);
        free(csr->val);
        free(csr);
    }
}

static CSRMatrix *coo_to_csr(const TSparseMatrix *m) {
    CSRMatrix *csr = malloc(sizeof(CSRMatrix));
    if (!csr) return NULL;

    unsigned int slots = m->nnz ? m->nnz : 1;      // never ask malloc for 0 bytes
    csr->nrows = m->nrows;
    csr->ncols = m->ncols;
    csr->rowptr = calloc((size_t)m->nrows + 1, sizeof(int));   // starts as all zeros
    csr->col = malloc(slots * sizeof(int));
    csr->val = malloc(slots * sizeof(double));
    int *next = malloc(((size_t)m->nrows + 1) * sizeof(int));
    if (!csr->rowptr || !csr->col || !csr->val || !next) {
        free(next);
        free_csr(csr);
        return NULL;
    }

    // step 1: count how many entries each row has
    for (unsigned int e = 0; e < m->nnz; e++) {
        int r = m->elements[e].r;
        int c = m->elements[e].c;
        if (r < 0 || (unsigned int)r >= m->nrows || c < 0 || (unsigned int)c >= m->ncols) {
            free(next);
            free_csr(csr);
            return NULL;
        }
        csr->rowptr[r + 1]++;
    }

    // step 2: running total, so rowptr[i] = where row i starts
    for (unsigned int i = 0; i < m->nrows; i++)
        csr->rowptr[i + 1] += csr->rowptr[i];

    // step 3: drop every entry into its row. next[i] = next free spot in row i
    memcpy(next, csr->rowptr, ((size_t)m->nrows + 1) * sizeof(int));
    for (unsigned int e = 0; e < m->nnz; e++) {
        int r = m->elements[e].r;
        int spot = next[r]++;
        csr->col[spot] = m->elements[e].c;
        csr->val[spot] = m->elements[e].val;
    }

    free(next);
    return csr;
}

/* A list of answer entries that grows as we add to it. */
typedef struct {
    COOElement *items;
    unsigned int count;
    unsigned int capacity;
} ResultList;

/* Add one entry to the list. Returns 0 if ok, -1 if we ran out of memory. */
static int result_add(ResultList *list, int r, int c, double v) {
    if (list->count == list->capacity) {
        unsigned int new_capacity = list->capacity ? list->capacity * 2 : 1024;
        COOElement *bigger = realloc(list->items, (size_t)new_capacity * sizeof(COOElement));
        if (!bigger) return -1;
        list->items = bigger;
        list->capacity = new_capacity;
    }
    list->items[list->count].r = r;
    list->items[list->count].c = c;
    list->items[list->count].val = v;
    list->count++;
    return 0;
}

/* Scratch space for working on one row. Every thread gets its OWN copy, which
 * is the trick that lets threads work at the same time without any locks. */
typedef struct {
    double *acc;     // acc[j] = running total for column j of the current row
    int *mark;       // mark[j] = which row last touched column j (-1 = nobody)
    int *cols;       // list of the columns touched in the current row
} Workspace;

static void workspace_free(Workspace *w) {
    free(w->acc);
    free(w->mark);
    free(w->cols);
    w->acc = NULL;
    w->mark = NULL;
    w->cols = NULL;
}

static int workspace_init(Workspace *w, unsigned int ncols) {
    size_t n = ncols ? ncols : 1;
    w->acc = malloc(n * sizeof(double));
    w->mark = malloc(n * sizeof(int));
    w->cols = malloc(n * sizeof(int));
    if (!w->acc || !w->mark || !w->cols) {
        workspace_free(w);
        return -1;
    }
    for (size_t j = 0; j < n; j++)
        w->mark[j] = -1;
    return 0;
}

/* Work out rows first_row .. last_row-1 of the answer and add them to "out".
 * Returns 0 if ok, -1 if we ran out of memory. */
static int multiply_rows(const CSRMatrix *A, const CSRMatrix *B,
                         int first_row, int last_row,
                         Workspace *w, ResultList *out) {
    for (int i = first_row; i < last_row; i++) {
        int touched = 0;    // how many different columns this row has hit so far

        // go through every entry a(i,k) in row i of A ...
        for (int p = A->rowptr[i]; p < A->rowptr[i + 1]; p++) {
            int k = A->col[p];
            double a = A->val[p];

            // ... and multiply it by every entry b(k,j) in row k of B
            for (int q = B->rowptr[k]; q < B->rowptr[k + 1]; q++) {
                int j = B->col[q];
                if (w->mark[j] != i) {
                    // first time this row lands on column j: start its total
                    w->mark[j] = i;
                    w->acc[j] = a * B->val[q];
                    w->cols[touched++] = j;
                } else {
                    // seen column j before in this row: add to its total
                    w->acc[j] += a * B->val[q];
                }
            }
        }

        // copy this row's totals into the answer
        for (int t = 0; t < touched; t++) {
            int j = w->cols[t];
            if (result_add(out, i, j, w->acc[j]) != 0)
                return -1;
        }
    }
    return 0;
}

/* Package up a finished list as a TSparseMatrix (it takes over the list's memory). */
static TSparseMatrix *make_result(unsigned int nrows, unsigned int ncols, ResultList *list) {
    TSparseMatrix *result = malloc(sizeof(TSparseMatrix));
    if (!result) return NULL;

    if (list->count == 0) {
        free(list->items);
        list->items = malloc(sizeof(COOElement));    // keep a valid pointer even if empty
        if (!list->items) {
            free(result);
            return NULL;
        }
    } else {
        // trim off the extra room we were keeping for growth (if it fails, no big deal)
        COOElement *trimmed = realloc(list->items, (size_t)list->count * sizeof(COOElement));
        if (trimmed) list->items = trimmed;
    }

    result->nrows = nrows;
    result->ncols = ncols;
    result->nnz = list->count;
    result->elements = list->items;
    list->items = NULL;
    return result;
}

/* ======================================================================
 * Sequential SpMM baseline (the "ground truth")
 * ====================================================================== */
TSparseMatrix *spmm_sequential(TSparseMatrix *A, TSparseMatrix *B) {
    if (A == NULL || B == NULL || A->ncols != B->nrows) return NULL;

    TSparseMatrix *result = NULL;
    CSRMatrix *csrA = coo_to_csr(A);
    CSRMatrix *csrB = coo_to_csr(B);
    Workspace w = {NULL, NULL, NULL};
    ResultList out = {NULL, 0, 0};

    if (csrA && csrB && workspace_init(&w, B->ncols) == 0) {
        // do every row of the answer, top to bottom
        if (multiply_rows(csrA, csrB, 0, (int)A->nrows, &w, &out) == 0)
            result = make_result(A->nrows, B->ncols, &out);
    }

    // clean up our scratch stuff (the answer itself is kept)
    free(out.items);
    workspace_free(&w);
    free_csr(csrA);
    free_csr(csrB);
    return result;
}

/* ======================================================================
 * Parallel SpMM
 *
 * Load balancing: rows of a sparse matrix can be wildly different sizes, so
 * giving each thread "the same number of rows" would be unfair. Instead:
 *   - For each row we work out how many multiplications it needs (its "cost").
 *   - We chop the rows into lots of small chunks that each cost about the
 *     same (about 1/16th of a thread's fair share).
 *   - Threads grab the next unclaimed chunk whenever they're free, like a
 *     line at a deli counter. Fast threads just end up taking more chunks.
 *
 * No race conditions: each chunk writes only into its OWN result list, and
 * each thread has its OWN scratch space. The only shared thing is the
 * "next chunk" counter, and we lock it once per chunk (never per
 * multiplication), so the locking costs next to nothing.
 *
 * It runs in two rounds of threads:
 *   Round 1: multiply (each chunk fills its own list).
 *   Round 2: copy the chunk lists into one big array, in chunk order, which
 *            gives exactly the same answer (in the same order) as the
 *            sequential version. We do this copying in parallel too, so it
 *            doesn't become the slow part.
 * ====================================================================== */

#define CHUNKS_PER_THREAD 16
#define MAX_THREADS 256

typedef struct {
    const CSRMatrix *A;
    const CSRMatrix *B;
    const int *chunk_start;       // chunk c covers rows chunk_start[c] .. chunk_start[c+1]-1
    int num_chunks;
    ResultList *chunk_results;    // one result list per chunk
    const unsigned int *offset;   // round 2: where chunk c's entries go in the big array
    COOElement *final_items;      // round 2: the big array
    int next_chunk;               // the "now serving" number (protected by lock)
    int failed;                   // becomes 1 if somebody runs out of memory (protected by lock)
    pthread_mutex_t lock;
} SharedWork;

/* How many multiplications does row i of the answer need? (+1 so empty rows still count a bit) */
static long long row_cost(const CSRMatrix *A, const CSRMatrix *B, int i) {
    long long cost = 1;
    for (int p = A->rowptr[i]; p < A->rowptr[i + 1]; p++) {
        int k = A->col[p];
        cost += B->rowptr[k + 1] - B->rowptr[k];
    }
    return cost;
}

/* Take a number: returns the next chunk to work on (a number >= num_chunks means "none left"). */
static int next_chunk_number(SharedWork *shared) {
    pthread_mutex_lock(&shared->lock);
    int c = shared->next_chunk++;
    if (shared->failed) c = shared->num_chunks;     // somebody had a problem, everyone stop
    pthread_mutex_unlock(&shared->lock);
    return c;
}

static void report_failure(SharedWork *shared) {
    pthread_mutex_lock(&shared->lock);
    shared->failed = 1;
    pthread_mutex_unlock(&shared->lock);
}

/* Round 1, what every thread runs: keep grabbing chunks and multiplying them. */
static void *multiply_worker(void *arg) {
    SharedWork *shared = (SharedWork *)arg;
    Workspace w;

    if (workspace_init(&w, shared->B->ncols) != 0) {
        report_failure(shared);
        return NULL;
    }

    int c;
    while ((c = next_chunk_number(shared)) < shared->num_chunks) {
        int ok = multiply_rows(shared->A, shared->B,
                               shared->chunk_start[c], shared->chunk_start[c + 1],
                               &w, &shared->chunk_results[c]);
        if (ok != 0) {
            report_failure(shared);
            break;
        }
    }

    workspace_free(&w);
    return NULL;
}

/* Round 2, what every thread runs: copy chunks into the big array. */
static void *copy_worker(void *arg) {
    SharedWork *shared = (SharedWork *)arg;

    int c;
    while ((c = next_chunk_number(shared)) < shared->num_chunks) {
        ResultList *mine = &shared->chunk_results[c];
        if (mine->count > 0)
            memcpy(shared->final_items + shared->offset[c], mine->items,
                   (size_t)mine->count * sizeof(COOElement));
        free(mine->items);          // this chunk's list isn't needed anymore
        mine->items = NULL;
    }
    return NULL;
}

/* Run "job" on num_workers threads and wait for all of them to finish.
 * The main thread is one of the workers, so we only start (num_workers - 1) more. */
static void run_threads(void *(*job)(void *), SharedWork *shared, int num_workers, pthread_t *threads) {
    shared->next_chunk = 0;         // start the "now serving" counter over

    int started = 0;
    for (int t = 1; t < num_workers; t++) {
        if (pthread_create(&threads[started], NULL, job, shared) != 0) {
            fprintf(stderr, "Warning: pthread_create() failed, using fewer threads.\n");
            break;      // fine, the threads we do have will cover all the chunks
        }
        started++;
    }

    job(shared);        // main thread does its share too

    for (int t = 0; t < started; t++)
        pthread_join(threads[t], NULL);
}

TSparseMatrix *spmm_parallel(TSparseMatrix *A, TSparseMatrix *B, int num_threads) {
    if (A == NULL || B == NULL || A->ncols != B->nrows || num_threads <= 0) return NULL;
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;

    TSparseMatrix *result = NULL;
    CSRMatrix *csrA = coo_to_csr(A);
    CSRMatrix *csrB = coo_to_csr(B);
    long long *cost = NULL;
    int *chunk_start = NULL;
    ResultList *chunk_results = NULL;
    unsigned int *offset = NULL;
    pthread_t *threads = NULL;
    COOElement *final_items = NULL;
    int num_chunks = 0;
    int lock_ready = 0;
    SharedWork shared;

    if (!csrA || !csrB)
        goto cleanup;               // (jump down to the clean-up code at the bottom)

    // ---- step 1: work out what each row costs, and the total ----
    cost = malloc((A->nrows ? A->nrows : 1) * sizeof(long long));
    if (!cost)
        goto cleanup;

    long long total = 0;
    for (unsigned int i = 0; i < A->nrows; i++) {
        cost[i] = row_cost(csrA, csrB, (int)i);
        total += cost[i];
    }

    long long wanted_chunks = (long long)num_threads * CHUNKS_PER_THREAD;
    long long target = total / wanted_chunks;      // cost we'd like in each chunk
    if (target < 1) target = 1;

    // ---- step 2: cut the rows into chunks ----
    // (we can never end up with more than 2*wanted_chunks+1 chunks)
    chunk_start = malloc((size_t)(2 * wanted_chunks + 3) * sizeof(int));
    if (!chunk_start)
        goto cleanup;

    chunk_start[0] = 0;
    long long running = 0;
    for (unsigned int i = 0; i < A->nrows; i++) {
        running += cost[i];
        if (running >= target) {                   // chunk is big enough, close it
            num_chunks++;
            chunk_start[num_chunks] = (int)i + 1;
            running = 0;
        }
    }
    if (chunk_start[num_chunks] != (int)A->nrows) { // leftover rows at the end
        num_chunks++;
        chunk_start[num_chunks] = (int)A->nrows;
    }

    chunk_results = calloc(num_chunks ? num_chunks : 1, sizeof(ResultList));
    offset = malloc((num_chunks ? num_chunks : 1) * sizeof(unsigned int));
    if (!chunk_results || !offset)
        goto cleanup;

    // ---- step 3: set up the shared info ----
    shared.A = csrA;
    shared.B = csrB;
    shared.chunk_start = chunk_start;
    shared.num_chunks = num_chunks;
    shared.chunk_results = chunk_results;
    shared.offset = offset;
    shared.final_items = NULL;
    shared.next_chunk = 0;
    shared.failed = 0;
    if (pthread_mutex_init(&shared.lock, NULL) != 0)
        goto cleanup;
    lock_ready = 1;

    int num_workers = num_threads;
    if (num_workers > num_chunks) num_workers = num_chunks;   // no point having more threads than chunks
    if (num_workers < 1) num_workers = 1;

    threads = malloc(num_workers * sizeof(pthread_t));
    if (!threads)
        goto cleanup;

    // ---- step 4: round 1, multiply ----
    run_threads(multiply_worker, &shared, num_workers, threads);
    if (shared.failed)
        goto cleanup;

    // ---- step 5: figure out where each chunk's entries go in the big array ----
    unsigned long long total_nnz = 0;
    for (int c = 0; c < num_chunks; c++) {
        if (total_nnz > UINT_MAX)
            goto cleanup;
        offset[c] = (unsigned int)total_nnz;      // chunk c starts after all the earlier chunks
        total_nnz += chunk_results[c].count;
    }
    if (total_nnz > UINT_MAX)
        goto cleanup;

    final_items = malloc((total_nnz ? (size_t)total_nnz : 1) * sizeof(COOElement));
    result = malloc(sizeof(TSparseMatrix));
    if (!final_items || !result) {
        free(result);
        result = NULL;
        goto cleanup;
    }
    shared.final_items = final_items;

    // ---- step 6: round 2, copy everything into the big array ----
    run_threads(copy_worker, &shared, num_workers, threads);

    result->nrows = A->nrows;
    result->ncols = B->ncols;
    result->nnz = (unsigned int)total_nnz;
    result->elements = final_items;
    final_items = NULL;             // the result owns it now, so don't free it below

cleanup:
    if (chunk_results) {
        for (int c = 0; c < num_chunks; c++)
            free(chunk_results[c].items);
        free(chunk_results);
    }
    if (lock_ready) pthread_mutex_destroy(&shared.lock);
    free(final_items);
    free(offset);
    free(threads);
    free(chunk_start);
    free(cost);
    free_csr(csrA);
    free_csr(csrB);
    return result;
}
