#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>
#include <stdarg.h>

#if ( (defined __linux__) || (defined __hpux))
#include <complex.h>
#endif 

#ifdef __sun
#include <complex>
 using namespace std;
#endif

#if ( (defined __linux__) || defined USE_OWN_BLAS_LAPACK )
/* BLAS I */ 
#define zcopy  zcopy_
#define dcopy  dcopy_
#define zaxpy  zaxpy_
#define daxpy  daxpy_
#define zdotc  zdotc_
#define zdotu  zdotu_
#define dznrm2 dznrm2_
#define dnrm2  dnrm2_
#define zdscal zdscal_
#define zscal  zscal_
#define idamax idamax_
/* BLAS II */ 
#define zgemv  zgemv_ 
#define ztrsv  ztrsv_

typedef complex<double> compdoub ;
extern "C" void zcopy(int *n, compdoub *dzx, int *incx, compdoub *dzy, int *incy);
extern "C" void dcopy(int *n, double* dx, int* incx, double* dy, int* incy);
extern "C" void zaxpy(int *n, compdoub *dza, compdoub *dzx,
                      int* incx, compdoub *dzy, int* incy);
extern "C" void daxpy(int *n, double *da, double *dx, int *incx, double *dy, 
                      int *incy); 
extern "C" compdoub zdotc(int *n, compdoub *dzx, int *incx, compdoub *dzy, 
                          int *incy);
extern "C" compdoub zdotu(int *n, compdoub *dzx, int *incx, compdoub *dzy, 
                          int *incy);
extern "C" double dznrm2(int *n, compdoub *dzx, int *incx);
extern "C" double dnrm2(int *n, double *dzx, int *incx);
extern "C" void   zdscal(int *n, double *da, compdoub *dzx, int *incx);
extern "C" void   zscal(int *n, compdoub *da, compdoub *dzx, int *incx);
extern "C" int    idamax(int *n, double *dx, int *incx);
extern "C" void   zgemv(char* trans, int* m, int* n, compdoub *zalpha, 
                        compdoub *za, int* lda, compdoub *zx, int* 
                        incx, compdoub *zbeta, compdoub *zy, int* 
                        incy); 
extern "C" void   ztrsv(char* uplo, char* trans, char* diag, int* n, 
                        compdoub *za, int* lda, compdoub *zx, 
                        int* incx); 
#endif

#if ( (defined __sun) && ( ! defined USE_OWN_BLAS_LAPACK ))
// typedef complex<double> compdoub ;
typedef complex<double> compdoub ;
/*..BLAS I subroutines: SUN PERF calls..*/
extern "C" void zcopy(int n, compdoub *dzx, int incx, compdoub *dzy, int incy);
extern "C" void dcopy(int n, double *dx, int incx, double *dy, int incy);
extern "C" void zaxpy(int n, compdoub* dza, compdoub *dzx,
                      int incx, compdoub *dzy, int incy);
extern "C" void daxpy(int n, double da, double *dx, int incx, double *dy, 
                      int incy); 
extern "C" compdoub *zdotc(int n, compdoub *dzx, int incx, compdoub *dzy, 
                          int incy);
extern "C" compdoub *zdotu(int n, compdoub *dzx, int incx, compdoub *dzy, 
                          int incy);
extern "C" double dznrm2(int n, compdoub *dzx, int incx);
extern "C" double dnrm2(int n, double *dzx, int incx);
extern "C" void   zdscal(int n, double da, compdoub *dzx, int incx);
extern "C" void   zscal(int n, compdoub* da, compdoub *dzx, int incx);
extern "C" int    idamax(int n, double *dx, int incx); 
extern "C" void   zgemv(char trans, int m, int n, compdoub *zalpha, 
                        compdoub *za, int lda, compdoub *zx, int 
                        incx, compdoub *zbeta, compdoub *zy, int 
                        incy); 
extern "C" void   ztrsv(char uplo, char trans, char diag, int n, 
                        compdoub *za, int lda, compdoub *zx, 
                        int incx); 
#endif 

void DCOPY(int *n, double *dx, int *incx, double *dy, int *incy); 
void ZCOPY(int *n, compdoub *dzx, int *incx, compdoub *dzy, 
                      int *incy); 
void DAXPY(int *n, double *da, double *dx, int *incx, double *dy, 
                      int *incy);
void ZAXPY(int *n, compdoub *dza, compdoub *dzx, int *incx, compdoub *dzy, 
                      int *incy);
compdoub ZDOTC(int *n, compdoub *dzx, int *incx, compdoub *dzy, 
                int *incy);
compdoub ZDOTU(int *n, compdoub *dzx, int *incx, compdoub *dzy, 
                int *incy);
double DZNRM2(int *n, compdoub *dzx, int *incx);
double DNRM2(int *n, double *dx, int *incx); 
void ZDSCAL(int *n, double *da, compdoub *dzx, int *incx);
void ZSCAL(int *n, compdoub *da, compdoub *dzx, int *incx);
int IDAMAX(int *n, double *dx, int *incx);
void ZGEMV(char *trans, int *m, int *n, compdoub *zalpha, 
           compdoub *za, int *lda, compdoub *zx, int 
           *incx, compdoub *zbeta, compdoub *zy, int *incy);
void ZTRSV(char uplo, char trans, char diag, int n, 
           compdoub *za, int lda, compdoub *zx, 
           int incx); 

# define linefile printf("Line %lu in file %s.\n",__LINE__,__FILE__);fflush(stdout) 

struct OPTIONS{ char*   PROBLEM_TYPE;
                char*   ELEM_TYPE; 
                double* ANIS_PARAM; 
                char*   KSP_TYPE; 
                int*    KSP_MAX_IT; 
                double* KSP_RTOL; 
                char*   PC_TYPE; 
                char*   SCALE;
                int*    DEBUG;
                char*   BASE; 
                char*   EXTENSION; 
                int*    SAMG_ITS_ON_FEM;};


#endif//GLOBAL_H

