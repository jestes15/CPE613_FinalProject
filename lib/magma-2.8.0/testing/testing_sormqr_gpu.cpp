/*
    -- MAGMA (version 2.8.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2024

       @author Mark Gates
       @generated from testing/testing_zunmqr_gpu.cpp, normal z -> s, Thu Mar 28 12:26:44 2024
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sormqr_gpu
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float Cnorm, error, work[1];
    float c_neg_one = MAGMA_S_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t mm, m, n, k, size, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t nb, ldc, lda, lwork, lwork_max, dt_size;
    float *C, *R, *A, *hwork, *tau;
    magmaFloat_ptr dC, dA, dT;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    // need slightly looser bound (60*eps instead of 30*eps) for some tests
    opts.tolerance = max( 60., opts.tolerance );
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    // test all combinations of input parameters
    magma_side_t  side [] = { MagmaLeft,       MagmaRight   };
    magma_trans_t trans[] = { MagmaTrans, MagmaNoTrans };

    printf("%%   M     N     K   side   trans   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R||_F / ||QC||_F\n");
    printf("%%==============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( int iside = 0; iside < 2; ++iside ) {
      for( int itran = 0; itran < 2; ++itran ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[itest];
            n = opts.nsize[itest];
            k = opts.ksize[itest];
            ldc = magma_roundup( m, opts.align );  // multiple of 32 by default
            // A is mm x k == m x k (left) or n x k (right)
            mm = (side[iside] == MagmaLeft ? m : n);
            nb  = magma_get_sgeqrf_nb( mm, k );
            lda = magma_roundup( mm, opts.align );  // multiple of 32 by default
            gflops = FLOPS_SORMQR( m, n, k, side[iside] ) / 1e9;
            
            if ( side[iside] == MagmaLeft && m < k ) {
                printf( "%5lld %5lld %5lld   %4c   %5c   skipping because side=left  and m < k\n",
                        (long long) m, (long long) n, (long long) k,
                        lapacke_side_const( side[iside] ),
                        lapacke_trans_const( trans[itran] ) );
                continue;
            }
            if ( side[iside] == MagmaRight && n < k ) {
                printf( "%5lld %5lld %5lld   %4c   %5c   skipping because side=right and n < k\n",
                        (long long) m, (long long) n, (long long) k,
                        lapacke_side_const( side[iside] ),
                        lapacke_trans_const( trans[itran] ) );
                continue;
            }
            
            if ( side[iside] == MagmaLeft ) {
                // side = left
                lwork_max = (m - k + nb)*(n + nb) + n*nb;
                dt_size = ( 2*min(m,k) + magma_roundup( max(m,n), 32) )*nb;
            }
            else {
                // side = right
                lwork_max = (n - k + nb)*(m + nb) + m*nb;
                dt_size = ( 2*min(n,k) + magma_roundup( max(m,n), 32 ) )*nb;
            }
            // this rounds it up slightly if needed to agree with lwork query below
            lwork_max = magma_int_t( real( magma_smake_lwork( lwork_max )));
            
            TESTING_CHECK( magma_smalloc_cpu( &C,     ldc*n ));
            TESTING_CHECK( magma_smalloc_cpu( &R,     ldc*n ));
            TESTING_CHECK( magma_smalloc_cpu( &A,     lda*k ));
            TESTING_CHECK( magma_smalloc_cpu( &hwork, lwork_max ));
            TESTING_CHECK( magma_smalloc_cpu( &tau,   k ));
            
            TESTING_CHECK( magma_smalloc( &dC, ldc*n ));
            TESTING_CHECK( magma_smalloc( &dA, lda*k ));
            TESTING_CHECK( magma_smalloc( &dT, dt_size ));
            
            // C is full, m x n
            size = ldc*n;
            lapackf77_slarnv( &ione, ISEED, &size, C );
            magma_ssetmatrix( m, n, C, ldc, dC, ldc, opts.queue );
            
            // A is mm x k
            magma_generate_matrix( opts, mm, k, A, lda );
            
            // compute QR factorization to get Householder vectors in dA, tau, dT
            magma_ssetmatrix( mm, k, A,  lda, dA, lda, opts.queue );
            magma_sgeqrf_gpu( mm, k, dA, lda, tau, dT, &info );
            magma_sgetmatrix( mm, k, dA, lda, A,  lda, opts.queue );
            if (info != 0) {
                printf("magma_sgeqrf_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_sormqr( lapack_side_const( side[iside] ), lapack_trans_const( trans[itran] ),
                              &m, &n, &k,
                              A, &lda, tau, C, &ldc, hwork, &lwork_max, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0) {
                printf("lapackf77_sormqr returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            // query for workspace size
            lwork = -1;
            magma_sormqr_gpu( side[iside], trans[itran],
                              m, n, k,
                              dA, lda, tau, dC, ldc, hwork, lwork, dT, nb, &info );
            if (info != 0) {
                printf("magma_sormqr_gpu (lwork query) returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            lwork = (magma_int_t) MAGMA_S_REAL( hwork[0] );
            if ( lwork < 0 || lwork > lwork_max  ) {
                printf("Warning: optimal lwork %lld > allocated lwork_max %lld\n", (long long) lwork, (long long) lwork_max );
                lwork = lwork_max;
            }
            
            // sormqr2 takes a copy of dA in CPU memory
            if ( opts.version == 2 ) {
                magma_sgetmatrix( mm, k, dA, lda, A, lda, opts.queue );
            }
            
            // TODO: sync still needed? on what queue?
            gpu_time = magma_sync_wtime( opts.queue );  // sync needed for L,N and R,T cases
            if ( opts.version == 1 ) {
                magma_sormqr_gpu( side[iside], trans[itran],
                                  m, n, k,
                                  dA, lda, tau, dC, ldc, hwork, lwork, dT, nb, &info );
            }
            else if ( opts.version == 2 ) {
                magma_sormqr2_gpu( side[iside], trans[itran],
                                   m, n, k,
                                   dA, lda, tau, dC, ldc, A, lda, &info );
            }
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_sormqr_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            magma_sgetmatrix( m, n, dC, ldc, R, ldc, opts.queue );
            
            /* =====================================================================
               compute relative error |QC_magma - QC_lapack| / |QC_lapack|
               =================================================================== */
            size = ldc*n;
            blasf77_saxpy( &size, &c_neg_one, C, &ione, R, &ione );
            Cnorm = lapackf77_slange( "Fro", &m, &n, C, &ldc, work );
            error = lapackf77_slange( "Fro", &m, &n, R, &ldc, work ) / (magma_ssqrt(m*n) * Cnorm);
            
            printf( "%5lld %5lld %5lld   %4c   %5c   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    (long long) m, (long long) n, (long long) k,
                    lapacke_side_const( side[iside] ),
                    lapacke_trans_const( trans[itran] ),
                    cpu_perf, cpu_time, gpu_perf, gpu_time,
                    error, (error < tol ? "ok" : "failed") );
            status += ! (error < tol);
            
            magma_free_cpu( C );
            magma_free_cpu( R );
            magma_free_cpu( A );
            magma_free_cpu( hwork );
            magma_free_cpu( tau );
            
            magma_free( dC );
            magma_free( dA );
            magma_free( dT );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }}  // end iside, itran
      printf( "\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
