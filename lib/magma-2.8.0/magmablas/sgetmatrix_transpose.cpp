/*
    -- MAGMA (version 2.8.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2024

       @generated from magmablas/zgetmatrix_transpose.cpp, normal z -> s, Thu Mar 28 12:27:58 2024

*/
#include "magma_internal.h"

/***************************************************************************//**
    Copy and transpose matrix dAT on GPU device to hA on CPU host.

    @param[in]  m       Number of rows    of output matrix hA. m >= 0.
    @param[in]  n       Number of columns of output matrix hA. n >= 0.
    @param[in]  nb      Block size. nb >= 0.
    @param[in]  dAT     The n-by-m matrix A^T on the GPU, of dimension (ldda,m).
    @param[in]  ldda    Leading dimension of matrix dAT. ldda >= n.
    @param[out] hA      The m-by-n matrix A on the CPU, of dimension (lda,n).
    @param[in]  lda     Leading dimension of matrix hA. lda >= m.
    @param[out] dwork   Workspace on the GPU, of dimension (2*lddw*nb).
    @param[in]  lddw    Leading dimension of dwork. lddw >= m.
    @param[in]  queues  Array of two queues, to pipeline operation.

    @ingroup magma_getmatrix_transpose
*******************************************************************************/
extern "C" void
magmablas_sgetmatrix_transpose(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloat_const_ptr dAT, magma_int_t ldda,
    float          *hA,  magma_int_t lda,
    magmaFloat_ptr       dwork,  magma_int_t lddw,
    magma_queue_t queues[2] )
{
#define    hA(i_, j_)    (hA + (i_) + (j_)*lda)
#define   dAT(i_, j_)   (dAT + (i_) + (j_)*ldda)
#define dwork(i_, j_) (dwork + (i_) + (j_)*lddw)

    magma_int_t i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    // TODO standard argument checking (xerbla)
    if (lda < m || ldda < n || lddw < m) {
        fprintf( stderr, "%s: wrong arguments.\n", __func__ );
        return;
    }

    for (i=0; i < n; i += nb) {
        /* Move data from GPU to CPU using 2 buffers; 1st transpose the data on the GPU */
        ib = min(n-i, nb);
        
        magmablas_stranspose( ib, m, dAT(i,0), ldda, dwork(0,(j%2)*nb), lddw, queues[j%2] );
        magma_sgetmatrix_async( m, ib,
                                dwork(0,(j%2)*nb), lddw,
                                hA(0,i), lda, queues[j%2] );
        j++;
    }
}
