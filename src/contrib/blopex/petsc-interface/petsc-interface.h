/* @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */
/* @@@ BLOPEX (version 1.1) LGPL Version 2.1 or above.See www.gnu.org. */
/* @@@ Copyright 2010 BLOPEX team http://code.google.com/p/blopex/     */
/* @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */

#ifndef PETSC_INTERFACE_HEADER
#define PETSC_INTERFACE_HEADER

#include "interpreter.h"

BlopexInt PETSC_dpotrf_interface (char *uplo, BlopexInt *n, double *a, BlopexInt * lda, BlopexInt *info);

BlopexInt PETSC_dsygv_interface (BlopexInt *itype, char *jobz, char *uplo, BlopexInt *
                    n, double *a, BlopexInt *lda, double *b, BlopexInt *ldb,
                    double *w, double *work, BlopexInt *lwork, BlopexInt *info);

BlopexInt PETSC_zpotrf_interface (char *uplo, BlopexInt *n, komplex *a, BlopexInt * lda, BlopexInt *info);

BlopexInt PETSC_zsygv_interface (BlopexInt *itype, char *jobz, char *uplo, BlopexInt *
                    n, komplex *a, BlopexInt *lda, komplex *b, BlopexInt *ldb,
                    double *w, komplex *work, BlopexInt *lwork, double *rwork, BlopexInt *info);

void *
PETSC_MimicVector( void *vvector );

BlopexInt
PETSC_DestroyVector( void *vvector );

BlopexInt
PETSC_InnerProd( void *x, void *y, void *result );

BlopexInt
PETSC_CopyVector( void *x, void *y );

BlopexInt
PETSC_ClearVector( void *x );

BlopexInt
PETSC_SetRandomValues( void* v, BlopexInt seed );

BlopexInt
PETSC_ScaleVector( void *alpha, void   *x);

BlopexInt
PETSC_Axpy( void *alpha,
                void   *x,
                void   *y );

int
LOBPCG_InitRandomContext(void);

int
LOBPCG_DestroyRandomContext(void);

int
PETSCSetupInterpreter( mv_InterfaceInterpreter *ii );

#endif /* PETSC_INTERFACE_HEADER */
