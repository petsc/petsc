/* This code was developed by Merico Argentati, Andrew Knyazev, Ilya Lashuk and Evgueni Ovtchinnikov */

#ifndef PETSC_INTERFACE_HEADER
#define PETSC_INTERFACE_HEADER

#include "interpreter.h"

#ifdef __cplusplus
extern "C" {
#endif

int PETSC_dpotrf_interface (char *uplo, int *n, double *a, int * lda, int *info);

int PETSC_dsygv_interface (int *itype, char *jobz, char *uplo, int *
                    n, double *a, int *lda, double *b, int *ldb,
                    double *w, double *work, int *lwork, int *info);

void *
PETSC_MimicVector( void *vvector );

int
PETSC_DestroyVector( void *vvector );

double
PETSC_InnerProd( void *x, void *y );

int
PETSC_CopyVector( void *x, void *y );

int
PETSC_ClearVector( void *x );

int
PETSC_SetRandomValues( void* v, int seed );

int
PETSC_ScaleVector( double  alpha, void   *x);

int
PETSC_Axpy( double alpha,
                void   *x,
                void   *y );

int
LOBPCG_InitRandomContext(void);

int 
LOBPCG_DestroyRandomContext(void);

int
PETSCSetupInterpreter( mv_InterfaceInterpreter *ii );

#ifdef __cplusplus
}
#endif

#endif /* PETSC_INTERFACE_HEADER */
