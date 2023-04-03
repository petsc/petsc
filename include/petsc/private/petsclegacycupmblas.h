#ifndef PETSCLEGACYCUPMBLAS_H
#define PETSCLEGACYCUPMBLAS_H

#include <petscdevice_cupm.h>

#if defined(PETSC_HAVE_CUDA)
  /* complex single */
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cublasXaxpy(a, b, c, d, e, f, g)                      cublasCaxpy((a), (b), (cuComplex *)(c), (cuComplex *)(d), (e), (cuComplex *)(f), (g))
      #define cublasXscal(a, b, c, d, e)                            cublasCscal((a), (b), (cuComplex *)(c), (cuComplex *)(d), (e))
      #define cublasXdotu(a, b, c, d, e, f, g)                      cublasCdotu((a), (b), (cuComplex *)(c), (d), (cuComplex *)(e), (f), (cuComplex *)(g))
      #define cublasXdot(a, b, c, d, e, f, g)                       cublasCdotc((a), (b), (cuComplex *)(c), (d), (cuComplex *)(e), (f), (cuComplex *)(g))
      #define cublasXswap(a, b, c, d, e, f)                         cublasCswap((a), (b), (cuComplex *)(c), (d), (cuComplex *)(e), (f))
      #define cublasXnrm2(a, b, c, d, e)                            cublasScnrm2((a), (b), (cuComplex *)(c), (d), (e))
      #define cublasIXamax(a, b, c, d, e)                           cublasIcamax((a), (b), (cuComplex *)(c), (d), (e))
      #define cublasXasum(a, b, c, d, e)                            cublasScasum((a), (b), (cuComplex *)(c), (d), (e))
      #define cublasXgemv(a, b, c, d, e, f, g, h, i, j, k, l)       cublasCgemv((a), (b), (c), (d), (cuComplex *)(e), (cuComplex *)(f), (g), (cuComplex *)(h), (i), (cuComplex *)(j), (cuComplex *)(k), (l))
      #define cublasXgemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n) cublasCgemm((a), (b), (c), (d), (e), (f), (cuComplex *)(g), (cuComplex *)(h), (i), (cuComplex *)(j), (k), (cuComplex *)(l), (cuComplex *)(m), (n))
      #define cublasXgeam(a, b, c, d, e, f, g, h, i, j, k, l, m)    cublasCgeam((a), (b), (c), (d), (e), (cuComplex *)(f), (cuComplex *)(g), (h), (cuComplex *)(i), (cuComplex *)(j), (k), (cuComplex *)(l), (m))
      #define cublasXgemvStridedBatched(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) \
        cublasCgemvStridedBatched((a), (b), (c), (d), (const cuComplex *)(e), (const cuComplex *)(f), (g), (h), (const cuComplex *)(i), (j), (k), (const cuComplex *)(l), (cuComplex *)(m), (n), (o), (p))
    #else /* complex double */
      #define cublasXaxpy(a, b, c, d, e, f, g)                      cublasZaxpy((a), (b), (cuDoubleComplex *)(c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (g))
      #define cublasXscal(a, b, c, d, e)                            cublasZscal((a), (b), (cuDoubleComplex *)(c), (cuDoubleComplex *)(d), (e))
      #define cublasXdotu(a, b, c, d, e, f, g)                      cublasZdotu((a), (b), (cuDoubleComplex *)(c), (d), (cuDoubleComplex *)(e), (f), (cuDoubleComplex *)(g))
      #define cublasXdot(a, b, c, d, e, f, g)                       cublasZdotc((a), (b), (cuDoubleComplex *)(c), (d), (cuDoubleComplex *)(e), (f), (cuDoubleComplex *)(g))
      #define cublasXswap(a, b, c, d, e, f)                         cublasZswap((a), (b), (cuDoubleComplex *)(c), (d), (cuDoubleComplex *)(e), (f))
      #define cublasXnrm2(a, b, c, d, e)                            cublasDznrm2((a), (b), (cuDoubleComplex *)(c), (d), (e))
      #define cublasIXamax(a, b, c, d, e)                           cublasIzamax((a), (b), (cuDoubleComplex *)(c), (d), (e))
      #define cublasXasum(a, b, c, d, e)                            cublasDzasum((a), (b), (cuDoubleComplex *)(c), (d), (e))
      #define cublasXgemv(a, b, c, d, e, f, g, h, i, j, k, l)       cublasZgemv((a), (b), (c), (d), (cuDoubleComplex *)(e), (cuDoubleComplex *)(f), (g), (cuDoubleComplex *)(h), (i), (cuDoubleComplex *)(j), (cuDoubleComplex *)(k), (l))
      #define cublasXgemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n) cublasZgemm((a), (b), (c), (d), (e), (f), (cuDoubleComplex *)(g), (cuDoubleComplex *)(h), (i), (cuDoubleComplex *)(j), (k), (cuDoubleComplex *)(l), (cuDoubleComplex *)(m), (n))
      #define cublasXgeam(a, b, c, d, e, f, g, h, i, j, k, l, m)    cublasZgeam((a), (b), (c), (d), (e), (cuDoubleComplex *)(f), (cuDoubleComplex *)(g), (h), (cuDoubleComplex *)(i), (cuDoubleComplex *)(j), (k), (cuDoubleComplex *)(l), (m))
      #define cublasXgemvStridedBatched(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) \
        cublasZgemvStridedBatched((a), (b), (c), (d), (const cuDoubleComplex *)(e), (const cuDoubleComplex *)(f), (g), (h), (const cuDoubleComplex *)(i), (j), (k), (const cuDoubleComplex *)(l), (cuDoubleComplex *)(m), (n), (o), (p))

    #endif
  #else /* real single */
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cublasXaxpy               cublasSaxpy
      #define cublasXscal               cublasSscal
      #define cublasXdotu               cublasSdot
      #define cublasXdot                cublasSdot
      #define cublasXswap               cublasSswap
      #define cublasXnrm2               cublasSnrm2
      #define cublasIXamax              cublasIsamax
      #define cublasXasum               cublasSasum
      #define cublasXgemv               cublasSgemv
      #define cublasXgemm               cublasSgemm
      #define cublasXgeam               cublasSgeam
      #define cublasXgemvStridedBatched cublasSgemvStridedBatched
    #else /* real double */
      #define cublasXaxpy               cublasDaxpy
      #define cublasXscal               cublasDscal
      #define cublasXdotu               cublasDdot
      #define cublasXdot                cublasDdot
      #define cublasXswap               cublasDswap
      #define cublasXnrm2               cublasDnrm2
      #define cublasIXamax              cublasIdamax
      #define cublasXasum               cublasDasum
      #define cublasXgemv               cublasDgemv
      #define cublasXgemm               cublasDgemm
      #define cublasXgeam               cublasDgeam
      #define cublasXgemvStridedBatched cublasDgemvStridedBatched
    #endif
  #endif

  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusolverDnXpotrf(a, b, c, d, e, f, g, h)                        cusolverDnCpotrf((a), (b), (c), (cuComplex *)(d), (e), (cuComplex *)(f), (g), (h))
      #define cusolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   cusolverDnCpotrf_bufferSize((a), (b), (c), (cuComplex *)(d), (e), (f))
      #define cusolverDnXpotrs(a, b, c, d, e, f, g, h, i)                     cusolverDnCpotrs((a), (b), (c), (d), (cuComplex *)(e), (f), (cuComplex *)(g), (h), (i))
      #define cusolverDnXpotri(a, b, c, d, e, f, g, h)                        cusolverDnCpotri((a), (b), (c), (cuComplex *)(d), (e), (cuComplex *)(f), (g), (h))
      #define cusolverDnXpotri_bufferSize(a, b, c, d, e, f)                   cusolverDnCpotri_bufferSize((a), (b), (c), (cuComplex *)(d), (e), (f))
      #define cusolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     cusolverDnCsytrf((a), (b), (c), (cuComplex *)(d), (e), (f), (cuComplex *)(g), (h), (i))
      #define cusolverDnXsytrf_bufferSize(a, b, c, d, e)                      cusolverDnCsytrf_bufferSize((a), (b), (cuComplex *)(c), (d), (e))
      #define cusolverDnXgetrf(a, b, c, d, e, f, g, h)                        cusolverDnCgetrf((a), (b), (c), (cuComplex *)(d), (e), (cuComplex *)(f), (g), (h))
      #define cusolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   cusolverDnCgetrf_bufferSize((a), (b), (c), (cuComplex *)(d), (e), (f))
      #define cusolverDnXgetrs(a, b, c, d, e, f, g, h, i, j)                  cusolverDnCgetrs((a), (b), (c), (d), (cuComplex *)(e), (f), (g), (cuComplex *)(h), (i), (j))
      #define cusolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   cusolverDnCgeqrf_bufferSize((a), (b), (c), (cuComplex *)(d), (e), (f))
      #define cusolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     cusolverDnCgeqrf((a), (b), (c), (cuComplex *)(d), (e), (cuComplex *)(f), (cuComplex *)(g), (h), (i))
      #define cusolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) cusolverDnCunmqr_bufferSize((a), (b), (c), (d), (e), (f), (cuComplex *)(g), (h), (cuComplex *)(i), (cuComplex *)(j), (k), (l))
      #define cusolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      cusolverDnCunmqr((a), (b), (c), (d), (e), (f), (cuComplex *)(g), (h), (cuComplex *)(i), (cuComplex *)(j), (k), (cuComplex *)(l), (m), (n))
      #define cublasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 cublasCtrsm((a), (b), (c), (d), (e), (f), (g), (cuComplex *)(h), (cuComplex *)(i), (j), (cuComplex *)(k), (l))
    #else /* complex double */
      #define cusolverDnXpotrf(a, b, c, d, e, f, g, h)                        cusolverDnZpotrf((a), (b), (c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (g), (h))
      #define cusolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   cusolverDnZpotrf_bufferSize((a), (b), (c), (cuDoubleComplex *)(d), (e), (f))
      #define cusolverDnXpotrs(a, b, c, d, e, f, g, h, i)                     cusolverDnZpotrs((a), (b), (c), (d), (cuDoubleComplex *)(e), (f), (cuDoubleComplex *)(g), (h), (i))
      #define cusolverDnXpotri(a, b, c, d, e, f, g, h)                        cusolverDnZpotri((a), (b), (c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (g), (h))
      #define cusolverDnXpotri_bufferSize(a, b, c, d, e, f)                   cusolverDnZpotri_bufferSize((a), (b), (c), (cuDoubleComplex *)(d), (e), (f))
      #define cusolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     cusolverDnZsytrf((a), (b), (c), (cuDoubleComplex *)(d), (e), (f), (cuDoubleComplex *)(g), (h), (i))
      #define cusolverDnXsytrf_bufferSize(a, b, c, d, e)                      cusolverDnZsytrf_bufferSize((a), (b), (cuDoubleComplex *)(c), (d), (e))
      #define cusolverDnXgetrf(a, b, c, d, e, f, g, h)                        cusolverDnZgetrf((a), (b), (c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (g), (h))
      #define cusolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   cusolverDnZgetrf_bufferSize((a), (b), (c), (cuDoubleComplex *)(d), (e), (f))
      #define cusolverDnXgetrs(a, b, c, d, e, f, g, h, i, j)                  cusolverDnZgetrs((a), (b), (c), (d), (cuDoubleComplex *)(e), (f), (g), (cuDoubleComplex *)(h), (i), (j))
      #define cusolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   cusolverDnZgeqrf_bufferSize((a), (b), (c), (cuDoubleComplex *)(d), (e), (f))
      #define cusolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     cusolverDnZgeqrf((a), (b), (c), (cuDoubleComplex *)(d), (e), (cuDoubleComplex *)(f), (cuDoubleComplex *)(g), (h), (i))
      #define cusolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) cusolverDnZunmqr_bufferSize((a), (b), (c), (d), (e), (f), (cuDoubleComplex *)(g), (h), (cuDoubleComplex *)(i), (cuDoubleComplex *)(j), (k), (l))
      #define cusolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      cusolverDnZunmqr((a), (b), (c), (d), (e), (f), (cuDoubleComplex *)(g), (h), (cuDoubleComplex *)(i), (cuDoubleComplex *)(j), (k), (cuDoubleComplex *)(l), (m), (n))
      #define cublasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 cublasZtrsm((a), (b), (c), (d), (e), (f), (g), (cuDoubleComplex *)(h), (cuDoubleComplex *)(i), (j), (cuDoubleComplex *)(k), (l))
    #endif
  #else /* real single */
    #if defined(PETSC_USE_REAL_SINGLE)
      #define cusolverDnXpotrf(a, b, c, d, e, f, g, h)                        cusolverDnSpotrf((a), (b), (c), (d), (e), (f), (g), (h))
      #define cusolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   cusolverDnSpotrf_bufferSize((a), (b), (c), (d), (e), (f))
      #define cusolverDnXpotrs(a, b, c, d, e, f, g, h, i)                     cusolverDnSpotrs((a), (b), (c), (d), (e), (f), (g), (h), (i))
      #define cusolverDnXpotri(a, b, c, d, e, f, g, h)                        cusolverDnSpotri((a), (b), (c), (d), (e), (f), (g), (h))
      #define cusolverDnXpotri_bufferSize(a, b, c, d, e, f)                   cusolverDnSpotri_bufferSize((a), (b), (c), (d), (e), (f))
      #define cusolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     cusolverDnSsytrf((a), (b), (c), (d), (e), (f), (g), (h), (i))
      #define cusolverDnXsytrf_bufferSize(a, b, c, d, e)                      cusolverDnSsytrf_bufferSize((a), (b), (c), (d), (e))
      #define cusolverDnXgetrf(a, b, c, d, e, f, g, h)                        cusolverDnSgetrf((a), (b), (c), (d), (e), (f), (g), (h))
      #define cusolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   cusolverDnSgetrf_bufferSize((a), (b), (c), (d), (e), (f))
      #define cusolverDnXgetrs(a, b, c, d, e, f, g, h, i, j)                  cusolverDnSgetrs((a), (b), (c), (d), (e), (f), (g), (h), (i), (j))
      #define cusolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   cusolverDnSgeqrf_bufferSize((a), (b), (c), (float *)(d), (e), (f))
      #define cusolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     cusolverDnSgeqrf((a), (b), (c), (float *)(d), (e), (float *)(f), (float *)(g), (h), (i))
      #define cusolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) cusolverDnSormqr_bufferSize((a), (b), (c), (d), (e), (f), (float *)(g), (h), (float *)(i), (float *)(j), (k), (l))
      #define cusolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      cusolverDnSormqr((a), (b), (c), (d), (e), (f), (float *)(g), (h), (float *)(i), (float *)(j), (k), (float *)(l), (m), (n))
      #define cublasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 cublasStrsm((a), (b), (c), (d), (e), (f), (g), (float *)(h), (float *)(i), (j), (float *)(k), (l))
    #else /* real double */
      #define cusolverDnXpotrf(a, b, c, d, e, f, g, h)                        cusolverDnDpotrf((a), (b), (c), (d), (e), (f), (g), (h))
      #define cusolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   cusolverDnDpotrf_bufferSize((a), (b), (c), (d), (e), (f))
      #define cusolverDnXpotrs(a, b, c, d, e, f, g, h, i)                     cusolverDnDpotrs((a), (b), (c), (d), (e), (f), (g), (h), (i))
      #define cusolverDnXpotri(a, b, c, d, e, f, g, h)                        cusolverDnDpotri((a), (b), (c), (d), (e), (f), (g), (h))
      #define cusolverDnXpotri_bufferSize(a, b, c, d, e, f)                   cusolverDnDpotri_bufferSize((a), (b), (c), (d), (e), (f))
      #define cusolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     cusolverDnDsytrf((a), (b), (c), (d), (e), (f), (g), (h), (i))
      #define cusolverDnXsytrf_bufferSize(a, b, c, d, e)                      cusolverDnDsytrf_bufferSize((a), (b), (c), (d), (e))
      #define cusolverDnXgetrf(a, b, c, d, e, f, g, h)                        cusolverDnDgetrf((a), (b), (c), (d), (e), (f), (g), (h))
      #define cusolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   cusolverDnDgetrf_bufferSize((a), (b), (c), (d), (e), (f))
      #define cusolverDnXgetrs(a, b, c, d, e, f, g, h, i, j)                  cusolverDnDgetrs((a), (b), (c), (d), (e), (f), (g), (h), (i), (j))
      #define cusolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   cusolverDnDgeqrf_bufferSize((a), (b), (c), (double *)(d), (e), (f))
      #define cusolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     cusolverDnDgeqrf((a), (b), (c), (double *)(d), (e), (double *)(f), (double *)(g), (h), (i))
      #define cusolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) cusolverDnDormqr_bufferSize((a), (b), (c), (d), (e), (f), (double *)(g), (h), (double *)(i), (double *)(j), (k), (l))
      #define cusolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      cusolverDnDormqr((a), (b), (c), (d), (e), (f), (double *)(g), (h), (double *)(i), (double *)(j), (k), (double *)(l), (m), (n))
      #define cublasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 cublasDtrsm((a), (b), (c), (d), (e), (f), (g), (double *)(h), (double *)(i), (j), (double *)(k), (l))
    #endif
  #endif
#endif // PETSC_HAVE_CUDA

#if defined(PETSC_HAVE_HIP)
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define hipsolverDnXpotrf(a, b, c, d, e, f, g, h)                        hipsolverCpotrf((a), (b), (c), (hipComplex *)(d), (e), (hipComplex *)(f), (g), (h))
      #define hipsolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   hipsolverCpotrf_bufferSize((a), (b), (c), (hipComplex *)(d), (e), (f))
      #define hipsolverDnXpotrs(a, b, c, d, e, f, g, h, i, j, k)               hipsolverCpotrs((a), (b), (c), (d), (hipComplex *)(e), (f), (hipComplex *)(g), (h), (hipComplex *)(i), (j), (k))
      #define hipsolverDnXpotri(a, b, c, d, e, f, g, h)                        hipsolverCpotri((a), (b), (c), (hipComplex *)(d), (e), (hipComplex *)(f), (g), (h))
      #define hipsolverDnXpotri_bufferSize(a, b, c, d, e, f)                   hipsolverCpotri_bufferSize((a), (b), (c), (hipComplex *)(d), (e), (f))
      #define hipsolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     hipsolverCsytrf((a), (b), (c), (hipComplex *)(d), (e), (f), (hipComplex *)(g), (h), (i))
      #define hipsolverDnXsytrf_bufferSize(a, b, c, d, e)                      hipsolverCsytrf_bufferSize((a), (b), (hipComplex *)(c), (d), (e))
      #define hipsolverDnXgetrf(a, b, c, d, e, f, g, h, i)                     hipsolverCgetrf((a), (b), (c), (hipComplex *)(d), (e), (hipComplex *)(f), (g), (h), (i))
      #define hipsolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   hipsolverCgetrf_bufferSize((a), (b), (c), (hipComplex *)(d), (e), (f))
      #define hipsolverDnXgetrs(a, b, c, d, e, f, g, h, i, j, k, l)            hipsolverCgetrs((a), (b), (c), (d), (hipComplex *)(e), (f), (g), (hipComplex *)(h), (i), (hipComplex *)(j), (k), (l))
      #define hipsolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   hipsolverCgeqrf_bufferSize((a), (b), (c), (hipComplex *)(d), (e), (f))
      #define hipsolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     hipsolverCgeqrf((a), (b), (c), (hipComplex *)(d), (e), (hipComplex *)(f), (hipComplex *)(g), (h), (i))
      #define hipsolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) hipsolverCunmqr_bufferSize((a), (b), (c), (d), (e), (f), (hipComplex *)(g), (h), (hipComplex *)(i), (hipComplex *)(j), (k), (l))
      #define hipsolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      hipsolverCunmqr((a), (b), (c), (d), (e), (f), (hipComplex *)(g), (h), (hipComplex *)(i), (hipComplex *)(j), (k), (hipComplex *)(l), (m), (n))
      #define hipblasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 hipblasCtrsm((a), (b), (c), (d), (e), (f), (g), (hipComplex *)(h), (hipComplex *)(i), (j), (hipComplex *)(k), (l))
    #else /* complex double */
      #define hipsolverDnXpotrf(a, b, c, d, e, f, g, h)                        hipsolverZpotrf((a), (b), (c), (hipDoubleComplex *)(d), (e), (hipDoubleComplex *)(f), (g), (h))
      #define hipsolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   hipsolverZpotrf_bufferSize((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
      #define hipsolverDnXpotrs(a, b, c, d, e, f, g, h, i, j, k)               hipsolverZpotrs((a), (b), (c), (d), (hipDoubleComplex *)(e), (f), (hipDoubleComplex *)(g), (h), (hipDoubleComplex *)(i), (j), (k))
      #define hipsolverDnXpotri(a, b, c, d, e, f, g, h)                        hipsolverZpotri((a), (b), (c), (hipDoubleComplex *)(d), (e), (hipDoubleComplex *)(f), (g), (h))
      #define hipsolverDnXpotri_bufferSize(a, b, c, d, e, f)                   hipsolverZpotri_bufferSize((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
      #define hipsolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     hipsolverZsytrf((a), (b), (c), (hipDoubleComplex *)(d), (e), (f), (hipDoubleComplex *)(g), (h), (i))
      #define hipsolverDnXsytrf_bufferSize(a, b, c, d, e)                      hipsolverZsytrf_bufferSize((a), (b), (hipDoubleComplex *)(c), (d), (e))
      #define hipsolverDnXgetrf(a, b, c, d, e, f, g, h, i)                     hipsolverZgetrf((a), (b), (c), (hipDoubleComplex *)(d), (e), (hipDoubleComplex *)(f), (g), (h), (i))
      #define hipsolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   hipsolverZgetrf_bufferSize((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
      #define hipsolverDnXgetrs(a, b, c, d, e, f, g, h, i, j, k, l)            hipsolverZgetrs((a), (b), (c), (d), (hipDoubleComplex *)(e), (f), (g), (hipDoubleComplex *)(h), (i), (hipDoubleComplex *)(j), (k), (l))
      #define hipsolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   hipsolverZgeqrf_bufferSize((a), (b), (c), (hipDoubleComplex *)(d), (e), (f))
      #define hipsolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     hipsolverZgeqrf((a), (b), (c), (hipDoubleComplex *)(d), (e), (hipDoubleComplex *)(f), (hipDoubleComplex *)(g), (h), (i))
      #define hipsolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) hipsolverZunmqr_bufferSize((a), (b), (c), (d), (e), (f), (hipDoubleComplex *)(g), (h), (hipDoubleComplex *)(i), (hipDoubleComplex *)(j), (k), (l))
      #define hipsolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      hipsolverZunmqr((a), (b), (c), (d), (e), (f), (hipDoubleComplex *)(g), (h), (hipDoubleComplex *)(i), (hipDoubleComplex *)(j), (k), (hipDoubleComplex *)(l), (m), (n))
      #define hipblasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 hipblasZtrsm((a), (b), (c), (d), (e), (f), (g), (hipDoubleComplex *)(h), (hipDoubleComplex *)(i), (j), (hipDoubleComplex *)(k), (l))
    #endif
  #else /* real single */
    #if defined(PETSC_USE_REAL_SINGLE)
      #define hipsolverDnXpotrf(a, b, c, d, e, f, g, h)                        hipsolverSpotrf((a), (b), (c), (float *)(d), (e), (float *)(f), (g), (h))
      #define hipsolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   hipsolverSpotrf_bufferSize((a), (b), (c), (float *)(d), (e), (f))
      #define hipsolverDnXpotrs(a, b, c, d, e, f, g, h, i, j, k)               hipsolverSpotrs((a), (b), (c), (d), (float *)(e), (f), (float *)(g), (h), (float *)(i), (j), (k))
      #define hipsolverDnXpotri(a, b, c, d, e, f, g, h)                        hipsolverSpotri((a), (b), (c), (float *)(d), (e), (float *)(f), (g), (h))
      #define hipsolverDnXpotri_bufferSize(a, b, c, d, e, f)                   hipsolverSpotri_bufferSize((a), (b), (c), (float *)(d), (e), (f))
      #define hipsolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     hipsolverSsytrf((a), (b), (c), (float *)(d), (e), (f), (float *)(g), (h), (i))
      #define hipsolverDnXsytrf_bufferSize(a, b, c, d, e)                      hipsolverSsytrf_bufferSize((a), (b), (float *)(c), (d), (e))
      #define hipsolverDnXgetrf(a, b, c, d, e, f, g, h, i)                     hipsolverSgetrf((a), (b), (c), (float *)(d), (e), (float *)(f), (g), (h), (i))
      #define hipsolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   hipsolverSgetrf_bufferSize((a), (b), (c), (float *)(d), (e), (f))
      #define hipsolverDnXgetrs(a, b, c, d, e, f, g, h, i, j, k, l)            hipsolverSgetrs((a), (b), (c), (d), (float *)(e), (f), (g), (float *)(h), (i), (float *)(j), (k), (l))
      #define hipsolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   hipsolverSgeqrf_bufferSize((a), (b), (c), (float *)(d), (e), (f))
      #define hipsolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     hipsolverSgeqrf((a), (b), (c), (float *)(d), (e), (float *)(f), (float *)(g), (h), (i))
      #define hipsolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) hipsolverSormqr_bufferSize((a), (b), (c), (d), (e), (f), (float *)(g), (h), (float *)(i), (float *)(j), (k), (l))
      #define hipsolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      hipsolverSormqr((a), (b), (c), (d), (e), (f), (float *)(g), (h), (float *)(i), (float *)(j), (k), (float *)(l), (m), (n))
      #define hipblasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 hipblasStrsm((a), (b), (c), (d), (e), (f), (g), (float *)(h), (float *)(i), (j), (float *)(k), (l))
    #else /* real double */
      #define hipsolverDnXpotrf(a, b, c, d, e, f, g, h)                        hipsolverDpotrf((a), (b), (c), (double *)(d), (e), (double *)(f), (g), (h))
      #define hipsolverDnXpotrf_bufferSize(a, b, c, d, e, f)                   hipsolverDpotrf_bufferSize((a), (b), (c), (double *)(d), (e), (f))
      #define hipsolverDnXpotrs(a, b, c, d, e, f, g, h, i, j, k)               hipsolverDpotrs((a), (b), (c), (d), (double *)(e), (f), (double *)(g), (h), (double *)(i), (j), (k))
      #define hipsolverDnXpotri(a, b, c, d, e, f, g, h)                        hipsolverDpotri((a), (b), (c), (double *)(d), (e), (double *)(f), (g), (h))
      #define hipsolverDnXpotri_bufferSize(a, b, c, d, e, f)                   hipsolverDpotri_bufferSize((a), (b), (c), (double *)(d), (e), (f))
      #define hipsolverDnXsytrf(a, b, c, d, e, f, g, h, i)                     hipsolverDsytrf((a), (b), (c), (double *)(d), (e), (f), (double *)(g), (h), (i))
      #define hipsolverDnXsytrf_bufferSize(a, b, c, d, e)                      hipsolverDsytrf_bufferSize((a), (b), (double *)(c), (d), (e))
      #define hipsolverDnXgetrf(a, b, c, d, e, f, g, h, i)                     hipsolverDgetrf((a), (b), (c), (double *)(d), (e), (double *)(f), (g), (h), (i))
      #define hipsolverDnXgetrf_bufferSize(a, b, c, d, e, f)                   hipsolverDgetrf_bufferSize((a), (b), (c), (double *)(d), (e), (f))
      #define hipsolverDnXgetrs(a, b, c, d, e, f, g, h, i, j, k, l)            hipsolverDgetrs((a), (b), (c), (d), (double *)(e), (f), (g), (double *)(h), (i), (double *)(j), (k), (l))
      #define hipsolverDnXgeqrf_bufferSize(a, b, c, d, e, f)                   hipsolverDgeqrf_bufferSize((a), (b), (c), (double *)(d), (e), (f))
      #define hipsolverDnXgeqrf(a, b, c, d, e, f, g, h, i)                     hipsolverDgeqrf((a), (b), (c), (double *)(d), (e), (double *)(f), (double *)(g), (h), (i))
      #define hipsolverDnXormqr_bufferSize(a, b, c, d, e, f, g, h, i, j, k, l) hipsolverDormqr_bufferSize((a), (b), (c), (d), (e), (f), (double *)(g), (h), (double *)(i), (double *)(j), (k), (l))
      #define hipsolverDnXormqr(a, b, c, d, e, f, g, h, i, j, k, l, m, n)      hipsolverDormqr((a), (b), (c), (d), (e), (f), (double *)(g), (h), (double *)(i), (double *)(j), (k), (double *)(l), (m), (n))
      #define hipblasXtrsm(a, b, c, d, e, f, g, h, i, j, k, l)                 hipblasDtrsm((a), (b), (c), (d), (e), (f), (g), (double *)(h), (double *)(i), (j), (double *)(k), (l))
    #endif
  #endif

  /* complex single */
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define hipblasXaxpy(a, b, c, d, e, f, g)                      hipblasCaxpy((a), (b), (hipblasComplex *)(c), (hipblasComplex *)(d), (e), (hipblasComplex *)(f), (g))
      #define hipblasXscal(a, b, c, d, e)                            hipblasCscal((a), (b), (hipblasComplex *)(c), (hipblasComplex *)(d), (e))
      #define hipblasXdotu(a, b, c, d, e, f, g)                      hipblasCdotu((a), (b), (hipblasComplex *)(c), (d), (hipblasComplex *)(e), (f), (hipblasComplex *)(g))
      #define hipblasXdot(a, b, c, d, e, f, g)                       hipblasCdotc((a), (b), (hipblasComplex *)(c), (d), (hipblasComplex *)(e), (f), (hipblasComplex *)(g))
      #define hipblasXswap(a, b, c, d, e, f)                         hipblasCswap((a), (b), (hipblasComplex *)(c), (d), (hipblasComplex *)(e), (f))
      #define hipblasXnrm2(a, b, c, d, e)                            hipblasScnrm2((a), (b), (hipblasComplex *)(c), (d), (e))
      #define hipblasIXamax(a, b, c, d, e)                           hipblasIcamax((a), (b), (hipblasComplex *)(c), (d), (e))
      #define hipblasXasum(a, b, c, d, e)                            hipblasScasum((a), (b), (hipblasComplex *)(c), (d), (e))
      #define hipblasXgemv(a, b, c, d, e, f, g, h, i, j, k, l)       hipblasCgemv((a), (b), (c), (d), (hipblasComplex *)(e), (hipblasComplex *)(f), (g), (hipblasComplex *)(h), (i), (hipblasComplex *)(j), (hipblasComplex *)(k), (l))
      #define hipblasXgemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n) hipblasCgemm((a), (b), (c), (d), (e), (f), (hipblasComplex *)(g), (hipblasComplex *)(h), (i), (hipblasComplex *)(j), (k), (hipblasComplex *)(l), (hipblasComplex *)(m), (n))
      #define hipblasXgeam(a, b, c, d, e, f, g, h, i, j, k, l, m)    hipblasCgeam((a), (b), (c), (d), (e), (hipblasComplex *)(f), (hipblasComplex *)(g), (h), (hipblasComplex *)(i), (hipblasComplex *)(j), (k), (hipblasComplex *)(l), (m))
    #else /* complex double */
      #define hipblasXaxpy(a, b, c, d, e, f, g) hipblasZaxpy((a), (b), (hipblasDoubleComplex *)(c), (hipblasDoubleComplex *)(d), (e), (hipblasDoubleComplex *)(f), (g))
      #define hipblasXscal(a, b, c, d, e)       hipblasZscal((a), (b), (hipblasDoubleComplex *)(c), (hipblasDoubleComplex *)(d), (e))
      #define hipblasXdotu(a, b, c, d, e, f, g) hipblasZdotu((a), (b), (hipblasDoubleComplex *)(c), (d), (hipblasDoubleComplex *)(e), (f), (hipblasDoubleComplex *)(g))
      #define hipblasXdot(a, b, c, d, e, f, g)  hipblasZdotc((a), (b), (hipblasDoubleComplex *)(c), (d), (hipblasDoubleComplex *)(e), (f), (hipblasDoubleComplex *)(g))
      #define hipblasXswap(a, b, c, d, e, f)    hipblasZswap((a), (b), (hipblasDoubleComplex *)(c), (d), (hipblasDoubleComplex *)(e), (f))
      #define hipblasXnrm2(a, b, c, d, e)       hipblasDznrm2((a), (b), (hipblasDoubleComplex *)(c), (d), (e))
      #define hipblasIXamax(a, b, c, d, e)      hipblasIzamax((a), (b), (hipblasDoubleComplex *)(c), (d), (e))
      #define hipblasXasum(a, b, c, d, e)       hipblasDzasum((a), (b), (hipblasDoubleComplex *)(c), (d), (e))
      #define hipblasXgemv(a, b, c, d, e, f, g, h, i, j, k, l) \
        hipblasZgemv((a), (b), (c), (d), (hipblasDoubleComplex *)(e), (hipblasDoubleComplex *)(f), (g), (hipblasDoubleComplex *)(h), (i), (hipblasDoubleComplex *)(j), (hipblasDoubleComplex *)(k), (l))
      #define hipblasXgemm(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
        hipblasZgemm((a), (b), (c), (d), (e), (f), (hipblasDoubleComplex *)(g), (hipblasDoubleComplex *)(h), (i), (hipblasDoubleComplex *)(j), (k), (hipblasDoubleComplex *)(l), (hipblasDoubleComplex *)(m), (n))
      #define hipblasXgeam(a, b, c, d, e, f, g, h, i, j, k, l, m) \
        hipblasZgeam((a), (b), (c), (d), (e), (hipblasDoubleComplex *)(f), (hipblasDoubleComplex *)(g), (h), (hipblasDoubleComplex *)(i), (hipblasDoubleComplex *)(j), (k), (hipblasDoubleComplex *)(l), (m))
    #endif
  #else /* real single */
    #if defined(PETSC_USE_REAL_SINGLE)
      #define hipblasXaxpy  hipblasSaxpy
      #define hipblasXscal  hipblasSscal
      #define hipblasXdotu  hipblasSdot
      #define hipblasXdot   hipblasSdot
      #define hipblasXswap  hipblasSswap
      #define hipblasXnrm2  hipblasSnrm2
      #define hipblasIXamax hipblasIsamax
      #define hipblasXasum  hipblasSasum
      #define hipblasXgemv  hipblasSgemv
      #define hipblasXgemm  hipblasSgemm
      #define hipblasXgeam  hipblasSgeam
    #else /* real double */
      #define hipblasXaxpy  hipblasDaxpy
      #define hipblasXscal  hipblasDscal
      #define hipblasXdotu  hipblasDdot
      #define hipblasXdot   hipblasDdot
      #define hipblasXswap  hipblasDswap
      #define hipblasXnrm2  hipblasDnrm2
      #define hipblasIXamax hipblasIdamax
      #define hipblasXasum  hipblasDasum
      #define hipblasXgemv  hipblasDgemv
      #define hipblasXgemm  hipblasDgemm
      #define hipblasXgeam  hipblasDgeam
    #endif
  #endif
#endif // PETSC_HAVE_HIP

#endif // PETSCLEGACYCUPMBLAS_H
