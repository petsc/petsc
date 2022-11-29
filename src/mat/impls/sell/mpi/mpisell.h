
#ifndef __MPISELL_H
  #define __MPISELL_H
#endif
#include <../src/mat/impls/sell/seq/sell.h>

typedef struct {
  Mat         A, B; /* local submatrices: A (diag part),
                                           B (off-diag part) */
  PetscMPIInt size; /* size of communicator */
  PetscMPIInt rank; /* rank of proc in communicator */

  /* The following variables are used for matrix assembly */
  PetscBool    donotstash;        /* PETSC_TRUE if off processor entries dropped */
  MPI_Request *send_waits;        /* array of send requests */
  MPI_Request *recv_waits;        /* array of receive requests */
  PetscInt     nsends, nrecvs;    /* numbers of sends and receives */
  PetscScalar *svalues, *rvalues; /* sending and receiving data */
  PetscInt     rmax;              /* maximum message length */
#if defined(PETSC_USE_CTABLE)
  PetscHMapI colmap;
#else
  PetscInt *colmap; /* local col number of off-diag col */
#endif
  PetscInt *garray; /* global index of all off-processor columns */

  /* The following variables are used for matrix-vector products */
  Vec        lvec; /* local vector */
  Vec        diag;
  VecScatter Mvctx;       /* scatter context for vector */
  PetscBool  roworiented; /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */
  PetscInt    *rowindices;   /* column indices for row */
  PetscScalar *rowvalues;    /* nonzero values in row */
  PetscBool    getrowactive; /* indicates MatGetRow(), not restored */

  PetscInt *ld; /* number of entries per row left of diagona block */
} Mat_MPISELL;

PETSC_EXTERN PetscErrorCode MatCreate_MPISELL(Mat);
PETSC_INTERN PetscErrorCode MatSetUpMultiply_MPISELL(Mat);

PETSC_INTERN PetscErrorCode MatDisAssemble_MPISELL(Mat);
PETSC_INTERN PetscErrorCode MatDuplicate_MPISELL(Mat, MatDuplicateOption, Mat *);

PETSC_INTERN PetscErrorCode MatDestroy_MPISELL_PtAP(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_MPISELL(Mat);

PETSC_INTERN PetscErrorCode MatSetValues_MPISELL(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_INTERN PetscErrorCode MatSetOption_MPISELL(Mat, MatOption, PetscBool);
PETSC_INTERN PetscErrorCode MatGetSeqNonzeroStructure_MPISELL(Mat, Mat *);

PETSC_INTERN PetscErrorCode MatSetFromOptions_MPISELL(Mat, PetscOptionItems *);
PETSC_INTERN PetscErrorCode MatMPISELLSetPreallocation_MPISELL(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);

PETSC_INTERN PetscErrorCode MatConvert_MPISELL_MPIAIJ(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPISELL(Mat, MatType, MatReuse, Mat *);

PETSC_INTERN PetscErrorCode MatSOR_MPISELL(Mat, Vec, PetscReal, MatSORType, PetscReal, PetscInt, PetscInt, Vec);

PETSC_INTERN PetscErrorCode MatCreateColmap_MPISELL_Private(Mat);
PETSC_INTERN PetscErrorCode MatDiagonalScaleLocal_MPISELL(Mat, Vec);
