/*
    -- MAGMA (version 2.8.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2024

       @author Mark Gates
       @author Azzam Haidar

       @generated from magmablas/zlacpy.cu, normal z -> d, Thu Mar 28 12:28:00 2024

*/
#include "magma_internal.h"

// To deal with really large matrices, this launchs multiple super blocks,
// each with up to 64K-1 x 64K-1 thread blocks, which is up to 4194240 x 4194240 matrix with BLK=64.
// CUDA architecture 2.0 limits each grid dimension to 64K-1.
// Instances arose for vectors used by sparse matrices with M > 4194240, though N is small.
const magma_int_t max_blocks = 65535;

// BLK_X and BLK_Y need to be equal for dlaset_q to deal with diag & offdiag
// when looping over super blocks.
// Formerly, BLK_X and BLK_Y could be different.
#define BLK_X 64
#define BLK_Y BLK_X

/******************************************************************************/
/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to dlaset, dlacpy, dlag2s, clag2z, dgeadd.
*/
static __device__
void dlacpy_full_device(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}


/******************************************************************************/
/*
    Similar to dlacpy_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to dlaset, dlacpy, zlat2c, clat2z.
*/
static __device__
void dlacpy_lower_device(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m && ind + BLK_X > iby ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n && ind >= iby+j; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}


/******************************************************************************/
/*
    Similar to dlacpy_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to dlaset, dlacpy, zlat2c, clat2z.
*/
static __device__
void dlacpy_upper_device(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < m && ind < iby + BLK_Y ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( ind <= iby+j ) {
                    dB[j*lddb] = dA[j*ldda];
                }
            }
        }
    }
}


/******************************************************************************/
/*
    kernel wrappers to call the device functions.
*/
__global__
void dlacpy_full_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_full_device(m, n, dA, ldda, dB, lddb);
}

__global__
void dlacpy_lower_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

__global__
void dlacpy_upper_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_upper_device(m, n, dA, ldda, dB, lddb);
}


/******************************************************************************/
/*
    kernel wrappers to call the device functions for the batched routine.
*/
__global__
void dlacpy_full_kernel_batched(
    int m, int n,
    double const * const *dAarray, magma_int_t Ai, magma_int_t Aj, int ldda,
    double **dBarray, magma_int_t Bi, magma_int_t Bj, int lddb )
{
    int batchid = blockIdx.z;
    dlacpy_full_device(
            m, n,
            dAarray[batchid] + Aj*ldda + Ai, ldda,
            dBarray[batchid] + Bj*lddb + Bi, lddb);
}

__global__
void dlacpy_lower_kernel_batched(
    int m, int n,
    double const * const *dAarray, magma_int_t Ai, magma_int_t Aj, int ldda,
    double **dBarray, magma_int_t Bi, magma_int_t Bj, int lddb )
{
    int batchid = blockIdx.z;
    dlacpy_lower_device(
            m, n,
            dAarray[batchid] + Aj*ldda + Ai, ldda,
            dBarray[batchid] + Bj*lddb + Bi, lddb);
}

__global__
void dlacpy_upper_kernel_batched(
    int m, int n,
    double const * const *dAarray, magma_int_t Ai, magma_int_t Aj, int ldda,
    double **dBarray, magma_int_t Bi, magma_int_t Bj, int lddb )
{
    int batchid = blockIdx.z;
    dlacpy_upper_device(
            m, n,
            dAarray[batchid] + Aj*ldda + Ai, ldda,
            dBarray[batchid] + Bj*lddb + Bi, lddb);
}

/******************************************************************************/
/*
    kernel wrappers to call the device functions for the vbatched routine.
*/
__global__
void dlacpy_full_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    double const * const *dAarray, magma_int_t* ldda,
    double **dBarray, magma_int_t* lddb )
{
    const int batchid = blockIdx.z;
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if(blockIdx.x >= magma_ceildiv(my_m, BLK_X)) return;
    if(blockIdx.y >= magma_ceildiv(my_n, BLK_Y)) return;

    dlacpy_full_device(my_m, my_n, dAarray[batchid], (int)ldda[batchid], dBarray[batchid], (int)lddb[batchid]);
}

__global__
void dlacpy_lower_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    double const * const *dAarray, magma_int_t* ldda,
    double **dBarray, magma_int_t* lddb )
{
    const int batchid = blockIdx.z;
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if(blockIdx.x >= magma_ceildiv(my_m, BLK_X)) return;
    if(blockIdx.y >= magma_ceildiv(my_n, BLK_Y)) return;

    dlacpy_lower_device(my_m, my_n, dAarray[batchid], (int)ldda[batchid], dBarray[batchid], (int)lddb[batchid]);
}

__global__
void dlacpy_upper_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    double const * const *dAarray, magma_int_t* ldda,
    double **dBarray, magma_int_t* lddb )
{
    const int batchid = blockIdx.z;
    const int my_m = (int)m[batchid];
    const int my_n = (int)n[batchid];
    if(blockIdx.x >= magma_ceildiv(my_m, BLK_X)) return;
    if(blockIdx.y >= magma_ceildiv(my_n, BLK_Y)) return;

    dlacpy_upper_device(my_m, my_n, dAarray[batchid], (int)ldda[batchid], dBarray[batchid], (int)lddb[batchid]);
}


/***************************************************************************//**
    Purpose
    -------
    DLACPY copies all or part of a two-dimensional matrix dA to another
    matrix dB.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
      -     = MagmaFull:       All of the matrix dA

    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.

    @param[in]
    dA      DOUBLE PRECISION array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    @param[out]
    dB      DOUBLE PRECISION array, dimension (LDDB,N)
            The M-by-N matrix dB.
            On exit, dB = dA in the locations specified by UPLO.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lacpy
*******************************************************************************/
extern "C" void
magmablas_dlacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb,
    magma_queue_t queue )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)

    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( m == 0 || n == 0 ) {
        return;
    }

    assert( BLK_X == BLK_Y );
    const magma_int_t super_NB = max_blocks*BLK_X;
    dim3 super_grid( magma_ceildiv( m, super_NB ), magma_ceildiv( n, super_NB ) );

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    magma_int_t mm, nn;
    if ( uplo == MagmaLower ) {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = magma_ceildiv( mm, BLK_X );
            for( unsigned int j=0; j < super_grid.y && j <= i; ++j ) {  // from left to diagonal
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = magma_ceildiv( nn, BLK_Y );
                if ( i == j ) {  // diagonal super block
                    dlacpy_lower_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else {           // off diagonal super block
                    dlacpy_full_kernel <<< grid, threads, 0, queue->cuda_stream() >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else if ( uplo == MagmaUpper ) {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = magma_ceildiv( mm, BLK_X );
            for( unsigned int j=i; j < super_grid.y; ++j ) {  // from diagonal to right
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = magma_ceildiv( nn, BLK_Y );
                if ( i == j ) {  // diagonal super block
                    dlacpy_upper_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else {           // off diagonal super block
                    dlacpy_full_kernel <<< grid, threads, 0, queue->cuda_stream() >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else {
        // TODO: use cudaMemcpy or cudaMemcpy2D ?
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = magma_ceildiv( mm, BLK_X );
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = magma_ceildiv( nn, BLK_Y );
                dlacpy_full_kernel <<< grid, threads, 0, queue->cuda_stream() >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
        }
    }
}

extern "C" void
magmablas_dlacpy_internal_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDouble_ptr             dBarray[], magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    dim3 threads( BLK_X, 1, 1 );
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( m, BLK_X ), magma_ceildiv( n, BLK_Y ), ibatch );

        if ( uplo == MagmaLower ) {
            dlacpy_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, dAarray+i, Ai, Aj, ldda, dBarray+i, Bi, Bj, lddb );
        }
        else if ( uplo == MagmaUpper ) {
            dlacpy_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, dAarray+i, Ai, Aj, ldda, dBarray+i, Bi, Bj, lddb );
        }
        else {
            dlacpy_full_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, n, dAarray+i, Ai, Aj, ldda, dBarray+i, Bi, Bj, lddb );
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    DLACPY_BATCHED copies all or part of each two-dimensional matrix
    dAarray[i] to matrix dBarray[i], for 0 <= i < batchcount.

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t
            Specifies the part of each matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of each matrix dA

    @param[in]
    m       INTEGER
            The number of rows of each matrix dA.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix dA.  N >= 0.

    @param[in]
    dAarray DOUBLE PRECISION* array, dimension (batchCount)
            Array of pointers to the matrices dA, where each dA is of dimension (LDDA,N).
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array dA.  LDDA >= max(1,M).

    @param[out]
    dBarray DOUBLE PRECISION* array, dimension (batchCount)
            Array of pointers to the matrices dB, where each dB is of dimension (LDDB,N).
            The M-by-N matrix dB.
            On exit, dB = dA in the locations specified by UPLO.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array dB.  LDDB >= max(1,M).

    @param[in]
    batchCount  Number of matrices in dAarray and dBarray.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lacpy_batched
*******************************************************************************/
extern "C" void
magmablas_dlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const dAarray[], magma_int_t ldda,
    magmaDouble_ptr             dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    else if ( batchCount < 0 )
        info = -8;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( m == 0 || n == 0 || batchCount == 0 ) {
        return;
    }

    magmablas_dlacpy_internal_batched(
    uplo, m, n,
    dAarray, 0, 0, ldda,
    dBarray, 0, 0, lddb,
    batchCount, queue );
}

/***************************************************************************//**
    Purpose
    -------
    DLACPY_VBATCHED copies all or part of each two-dimensional matrix
    dAarray[i] to matrix dBarray[i], for 0 <= i < batchcount.
    Matrices are assumed to generally have different sizes/leading dimensions

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t
            Specifies the part of each matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of each matrix dA

    @param[in]
    max_m   INTEGER.
            maximum value of m.

    @param[in]
    max_n   INTEGER.
            maximum value of n.

    @param[in]
    m       INTEGER array, dimension (batchCount).
            Each is the number of rows of each matrix dA.  M >= 0.

    @param[in]
    n       INTEGER array, dimension (batchCount).
            The number of columns of each matrix dA.  N >= 0.

    @param[in]
    dAarray Array of pointers , dimension (batchCount)
            Each is a DOUBLE PRECISION array dA, where the ith matrix dA is of dimension (ldda[i],n[i]).
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.

    @param[in]
    ldda    INTEGER array, dimension (batchCount).
            Each is the leading dimension of each array dA. For the ith matrix dA ldda[i] >= max(1,m[i]).

    @param[out]
    dBarray Array of pointers, dimension(batchCount).
            Each is a DOUBLE PRECISION array dB, where the ith matrix dB is of dimension (lddb[i],n[i]).
            The M-by-N matrix dB.
            On exit, dB = dA in the locations specified by UPLO.

    @param[in]
    lddb    INTEGER array, dimension (batchCount).
            Each is the leading dimension of each array dB. For the ith matrix dB lddb[i] >= max(1,m[i]).

    @param[in]
    batchCount  Number of matrices in dAarray and dBarray.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lacpy_batched
*******************************************************************************/
extern "C" void
magmablas_dlacpy_vbatched(
    magma_uplo_t uplo,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t* m, magma_int_t* n,
    double const * const * dAarray, magma_int_t* ldda,
    double**               dBarray, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    //else if ( m < 0 )
    //    info = -2;
    //else if ( n < 0 )
    //    info = -3;
    //else if ( ldda < max(1,m))
    //    info = -5;
    //else if ( lddb < max(1,m))
    //    info = -7;
    else if ( batchCount < 0 )
        info = -8;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( max_m == 0 || max_n == 0 || batchCount == 0 ) {
        return;
    }

    dim3 threads( BLK_X, 1, 1 );

    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( max_m, BLK_X ), magma_ceildiv( max_n, BLK_Y ), ibatch );

        if ( uplo == MagmaLower ) {
            dlacpy_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>
            ( m+i, n+i, dAarray+i, ldda+i, dBarray+i, lddb+i );
        }
        else if ( uplo == MagmaUpper ) {
            dlacpy_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>
            ( m+i, n+i, dAarray+i, ldda+i, dBarray+i, lddb+i );
        }
        else {
            dlacpy_full_kernel_vbatched <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m+i, n+i, dAarray+i, ldda+i, dBarray+i, lddb+i );
        }
    }
}
