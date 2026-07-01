#pragma once

#include <petsc/private/matimpl.h>
#include <petscblaslapack.h>

typedef struct {
  PetscBLASInt ictxt;            /* process grid context */
  PetscBLASInt nprow, npcol;     /* number of process rows and columns */
  PetscBLASInt myrow, mycol;     /* coordinates of local process on the grid */
  PetscInt     grid_refct;       /* reference count */
  PetscBLASInt ictxrow, ictxcol; /* auxiliary 1d process grid contexts */
} Mat_ScaLAPACK_Grid;

typedef struct {
  Mat_ScaLAPACK_Grid *grid;        /* process grid */
  PetscBLASInt        desc[9];     /* ScaLAPACK descriptor */
  PetscBLASInt        M, N;        /* global dimensions, for rows and columns */
  PetscBLASInt        locr, locc;  /* dimensions of local array */
  PetscBLASInt        mb, nb;      /* block size, for rows and columns */
  PetscBLASInt        rsrc, csrc;  /* coordinates of process owning first row and column */
  PetscScalar        *loc;         /* pointer to local array */
  PetscBLASInt        lld;         /* local leading dimension */
  PetscBLASInt       *pivots;      /* pivots in LU factorization */
  PetscBool           roworiented; /* if true, row-oriented input (default) */
} Mat_ScaLAPACK;

PETSC_INTERN PetscErrorCode MatMatMultSymbolic_ScaLAPACK(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_ScaLAPACK(Mat, Mat, Mat);

/*MC
  PetscCallScaLAPACKInfo - Calls a ScaLAPACK routine that has an `info` return flag so that the stack trace returned from any signal received
                        inside the function call includes the name of the ScaLAPACK routine. Also checks the `info` flag.

  Synopsis:
  #include <petscscalapack.h>
  void PetscCallScaLAPACKInfo(char *name, routine)

  Not Collective

  Input Parameters:
+  name    - string that gives the name of the function being called
-  routine - actual call to the routine including its arguments

  Level: developer

  Notes:
  The calling routine should not declare the argument `info` but must call it `info` in the calling sequence.

  In a small number of cases, calls to ScaLAPACK routines that return a positive `info` code are allowed without ending the PETSc program.
  In these situations the macro `PetscCallBLAS()` should be used, not `PetscCallScaLAPACKInfo()`.

.seealso: `PetscCall()`, `PetscCallBLAS()`, `PetscCallLAPACKInfo()`
M*/
#define PetscCallScaLAPACKInfo(name, routine) \
  do { \
    PetscBLASInt info = -1000; \
    PetscCallBLAS(name, routine); \
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ScaLAPACK routine %s %" PetscBLASInt_FMT "-th argument had an illegal value", name, -info); \
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ScaLAPACK routine %s failed with INFO = %" PetscBLASInt_FMT ". Search at https://www.netlib.org/scalapack/explore-html/index.html to decode the error code.", name, info); \
  } while (0)

#define PETSC_PASTE4_(a, b, c, d) a##b##c##d
#define PETSC_PASTE4(a, b, c, d)  PETSC_PASTE4_(a, b, c, d)

#if defined(PETSC_BLASLAPACK_CAPS)
  #define PETSC_SCALAPACK_PREFIX_ P
  #define PETSCBLASNOTYPE(x, X)   PETSC_PASTE2(X, PETSC_BLASLAPACK_SUFFIX_)
  #define PETSCSCALAPACK(x, X)    PETSC_PASTE4(PETSC_SCALAPACK_PREFIX_, PETSC_BLASLAPACK_PREFIX_, X, PETSC_BLASLAPACK_SUFFIX_)
#else
  #define PETSC_SCALAPACK_PREFIX_ p
  #define PETSCBLASNOTYPE(x, X)   PETSC_PASTE2(x, PETSC_BLASLAPACK_SUFFIX_)
  #define PETSCSCALAPACK(x, X)    PETSC_PASTE4(PETSC_SCALAPACK_PREFIX_, PETSC_BLASLAPACK_PREFIX_, x, PETSC_BLASLAPACK_SUFFIX_)
#endif

/* BLACS routines (C interface) */
BLAS_EXTERN PetscBLASInt Csys2blacs_handle(MPI_Comm syscontext);
BLAS_EXTERN void         Cblacs_pinfo(PetscBLASInt *mypnum, PetscBLASInt *nprocs);
BLAS_EXTERN void         Cblacs_get(PetscBLASInt context, PetscBLASInt request, PetscBLASInt *value);
BLAS_EXTERN PetscBLASInt Cblacs_pnum(PetscBLASInt context, PetscBLASInt prow, PetscBLASInt pcol);
BLAS_EXTERN PetscBLASInt Cblacs_gridinit(PetscBLASInt *context, const char *order, PetscBLASInt np_row, PetscBLASInt np_col);
BLAS_EXTERN void         Cblacs_gridinfo(PetscBLASInt context, PetscBLASInt *np_row, PetscBLASInt *np_col, PetscBLASInt *my_row, PetscBLASInt *my_col);
BLAS_EXTERN void         Cblacs_gridexit(PetscBLASInt context);
BLAS_EXTERN void         Cblacs_exit(PetscBLASInt error_code);
BLAS_EXTERN void         Cdgebs2d(PetscBLASInt ctxt, const char *scope, const char *top, PetscBLASInt m, PetscBLASInt n, PetscScalar *A, PetscBLASInt lda);
BLAS_EXTERN void         Cdgebr2d(PetscBLASInt ctxt, const char *scope, const char *top, PetscBLASInt m, PetscBLASInt n, PetscScalar *A, PetscBLASInt lda, PetscBLASInt rsrc, PetscBLASInt csrc);
BLAS_EXTERN void         Cdgsum2d(PetscBLASInt ctxt, const char *scope, const char *top, PetscBLASInt m, PetscBLASInt n, PetscScalar *A, PetscBLASInt lda, PetscBLASInt rsrc, PetscBLASInt csrc);

/* PBLAS */
#define PBLASgemv_ PETSCSCALAPACK(gemv, GEMV)
#define PBLASgemm_ PETSCSCALAPACK(gemm, GEMM)
#if defined(PETSC_USE_COMPLEX)
  #define PBLAStran_ PETSCSCALAPACK(tranc, TRANC)
#else
  #define PBLAStran_ PETSCSCALAPACK(tran, TRAN)
#endif

BLAS_EXTERN void PBLASgemv_(const char *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void PBLASgemm_(const char *, const char *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void PBLAStran_(PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);

/* ScaLAPACK */
#define SCALAPACKlange_ PETSCSCALAPACK(lange, LANGE)
#define SCALAPACKpotrf_ PETSCSCALAPACK(potrf, POTRF)
#define SCALAPACKpotrs_ PETSCSCALAPACK(potrs, POTRS)
#define SCALAPACKgetrf_ PETSCSCALAPACK(getrf, GETRF)
#define SCALAPACKgetrs_ PETSCSCALAPACK(getrs, GETRS)

BLAS_EXTERN PetscReal SCALAPACKlange_(const char *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *);
BLAS_EXTERN void      SCALAPACKpotrf_(const char *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void      SCALAPACKpotrs_(const char *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void      SCALAPACKgetrf_(PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void      SCALAPACKgetrs_(const char *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);

/* auxiliary routines */
#define SCALAPACKnumroc_   PETSCBLASNOTYPE(numroc, NUMROC)
#define SCALAPACKdescinit_ PETSCBLASNOTYPE(descinit, DESCINIT)
#define SCALAPACKinfog2l_  PETSCBLASNOTYPE(infog2l, INFOG2L)
#define SCALAPACKgemr2d_   PETSCSCALAPACK(gemr2d, GEMR2D)
#define SCALAPACKmatadd_   PETSCSCALAPACK(matadd, MATADD)
#define SCALAPACKelset_    PETSCSCALAPACK(elset, ELSET)
#define SCALAPACKelget_    PETSCSCALAPACK(elget, ELGET)

BLAS_EXTERN PetscBLASInt SCALAPACKnumroc_(PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void         SCALAPACKdescinit_(PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void         SCALAPACKinfog2l_(PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void         SCALAPACKgemr2d_(PetscBLASInt *, PetscBLASInt *, const PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void         SCALAPACKmatadd_(PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
BLAS_EXTERN void         SCALAPACKelset_(PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *);
BLAS_EXTERN void         SCALAPACKelget_(const char *, const char *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);

/*
    Macros to test valid arguments
*/
#if !defined(PETSC_USE_DEBUG)

  #define MatScaLAPACKCheckDistribution(a, arga, b, argb) \
    do { \
      (void)(a); \
      (void)(b); \
    } while (0)

#else

  #define MatScaLAPACKCheckDistribution(a, arga, b, argb) \
    do { \
      Mat_ScaLAPACK *_aa = (Mat_ScaLAPACK *)(a)->data, *_bb = (Mat_ScaLAPACK *)(b)->data; \
      PetscCheck(_aa->mb == _bb->mb && _aa->nb == _bb->nb && _aa->rsrc == _bb->rsrc && _aa->csrc == _bb->csrc && _aa->grid->nprow == _bb->grid->nprow && _aa->grid->npcol == _bb->grid->npcol && _aa->grid->myrow == _bb->grid->myrow && \
                   _aa->grid->mycol == _bb->grid->mycol, \
                 PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_INCOMP, "Arguments #%d and #%d have different ScaLAPACK distribution", arga, argb); \
    } while (0)

#endif
