/*
    -- MAGMA (version 2.8.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2024

       @generated from testing/testing_zher2k.cpp, normal z -> d, Thu Mar 28 12:26:28 2024
       @author Chongxiao Cao
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define REAL


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dsyr2k
*/
int main( int argc, char** argv)
{
    #ifdef MAGMA_HAVE_OPENCL
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dB(i_, j_)  dB, ((i_) + (j_)*lddb)
    #define dC(i_, j_)  dC, ((i_) + (j_)*lddc)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dC(i_, j_) (dC + (i_) + (j_)*lddc)
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, dev_perf, dev_time, cpu_perf, cpu_time;
    double          magma_error, dev_error, work[1];
    magma_int_t N, K;
    magma_int_t Ak, An, Bk, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    double *hA, *hB, *hC, *hCmagma, *hCdev;
    magmaDouble_ptr dA, dB, dC;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // See testing_dgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    #ifdef COMPLEX
    if (opts.transA == MagmaTrans) {
        opts.transA = MagmaConjTrans; 
        printf("%% WARNING: transA = MagmaTrans changed to MagmaConjTrans\n");
    }
    #endif

    // We can check MAGMA vs. vendor
    printf("%% If running lapack (option --lapack), MAGMA and %s errors are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to %s result.\n\n",
            g_platform_str, g_platform_str );
        
    printf("%% uplo = %s, transA = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA) );
    printf("%%   N     K   MAGMA Gflop/s (ms)  %s Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error   %s error\n",
           g_platform_str, g_platform_str);

    printf("%%===================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.msize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_DSYR2K(K, N) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                Ak = K;
                ldb = Bn = N;
                Bk = K;
            } else {
                lda = An = K;
                Ak = N;
                ldb = Bn = K;
                Bk = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak;
            sizeB = ldb*Ak;
            sizeC = ldc*N;
            
            TESTING_CHECK( magma_dmalloc_cpu( &hA,      lda*Ak ));
            TESTING_CHECK( magma_dmalloc_cpu( &hB,      ldb*Bk ));
            TESTING_CHECK( magma_dmalloc_cpu( &hC,      ldc*N  ));
            TESTING_CHECK( magma_dmalloc_cpu( &hCmagma, ldc*N  ));
            TESTING_CHECK( magma_dmalloc_cpu( &hCdev,   ldc*N  ));
            
            TESTING_CHECK( magma_dmalloc( &dA, ldda*Ak ));
            TESTING_CHECK( magma_dmalloc( &dB, lddb*Bk ));
            TESTING_CHECK( magma_dmalloc( &dC, lddc*N  ));
            
            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_dlarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_dlarnv( &ione, ISEED, &sizeC, hC );
            
            // for error checks
            double Anorm = lapackf77_dlange( "F", &An, &Ak, hA, &lda, work );
            double Bnorm = lapackf77_dlange( "F", &Bn, &Bk, hB, &ldb, work );
            double Cnorm = safe_lapackf77_dlansy( "F", lapack_uplo_const(opts.uplo), &N, hC, &ldc, work );
            
            
            magma_dsetmatrix( An, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_dsetmatrix( Bn, Bk, hB, ldb, dB(0,0), lddb, opts.queue );
            /* =====================================================================
               Performs operation using MAGMABLAS ( for CUDA and HIP)
               =================================================================== */
            magma_dsetmatrix( N, N, hC, ldc, dC(0,0), lddc, opts.queue );
            
            magma_time = magma_sync_wtime( opts.queue );
            magmablas_dsyr2k( opts.uplo, opts.transA, N, K,
                              alpha, dA(0,0), ldda,
                                     dB(0,0), lddb,
                              beta,  dC(0,0), lddc, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_perf = gflops / magma_time;

            magma_dgetmatrix( N, N, dC(0,0), lddc, hCmagma, ldc, opts.queue );

           /* =====================================================================
               Performs operation using cuBLAS / clBLAS / hipBLAS
               =================================================================== */
            magma_dsetmatrix( N, N, hC, ldc, dC(0,0), lddc, opts.queue );
            
            dev_time = magma_sync_wtime( opts.queue );
            magma_dsyr2k( opts.uplo, opts.transA, N, K,
                          alpha, dA(0,0), ldda,
                                 dB(0,0), lddb,
                          beta,  dC(0,0), lddc, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_dgetmatrix( N, N, dC(0,0), lddc, hCdev, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_dsyr2k( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), &N, &K,
                               &alpha, hA, &lda,
                                       hB, &ldb,
                               &beta,  hC, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // See testing_dgemm for formula.
                blasf77_daxpy( &sizeC, &c_neg_one, hC, &ione, hCmagma, &ione );
                magma_error = safe_lapackf77_dlansy( "F", lapack_uplo_const(opts.uplo), &N, hCmagma, &ldc, work )
                            / (2*sqrt(double(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                blasf77_daxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = safe_lapackf77_dlansy( "F", lapack_uplo_const(opts.uplo), &N, hCdev, &ldc, work )
                            / (2*sqrt(double(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                bool okay = (magma_error < tol && dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %7.2f (%7.2f)   %8.2e      %8.2e   %s\n",
                       (long long) N, (long long) K,
                       magma_perf, 1000.*magma_time,
                       dev_perf,   1000.*dev_time,
                       cpu_perf,   1000.*cpu_time,
                       magma_error, dev_error, (okay ? "ok" : "failed"));
            }
            else {
                blasf77_daxpy( &sizeC, &c_neg_one, hCdev, &ione, hCmagma, &ione );
                magma_error = safe_lapackf77_dlansy( "F", lapack_uplo_const(opts.uplo), &N, hCmagma, &ldc, work )
                            / (2*sqrt(double(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)     ---   (  ---  )    %8.2e         ---      %s\n",
                       (long long) N, (long long) K,
                       magma_perf, 1000.*magma_time,
                       dev_perf,   1000.*dev_time,
                       magma_error, (okay ? "ok" : "failed"));                
            }
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            magma_free_cpu( hCdev );
            magma_free_cpu( hCmagma );
            
            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
