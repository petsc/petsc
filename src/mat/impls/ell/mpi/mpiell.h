
#if !defined(__MPIELL_H)
#define __MPIELL_H
#endif
#include <../src/mat/impls/ell/seq/ell.h>

typedef struct {
  Mat A,B;                             /* local submatrices: A (diag part),
                                           B (off-diag part) */
  PetscMPIInt size;                     /* size of communicator */
  PetscMPIInt rank;                     /* rank of proc in communicator */

  /* The following variables are used for matrix assembly */
  PetscBool   donotstash;               /* PETSC_TRUE if off processor entries dropped */
  MPI_Request *send_waits;              /* array of send requests */
  MPI_Request *recv_waits;              /* array of receive requests */
  PetscInt    nsends,nrecvs;           /* numbers of sends and receives */
  PetscScalar *svalues,*rvalues;       /* sending and receiving data */
  PetscInt    rmax;                     /* maximum message length */
#if defined(PETSC_USE_CTABLE)
  PetscTable colmap;
#else
  PetscInt *colmap;                     /* local col number of off-diag col */
#endif
  PetscInt *garray;                     /* global index of all off-processor columns */

  /* The following variables are used for matrix-vector products */
  Vec        lvec;                 /* local vector */
  Vec        diag;
  VecScatter Mvctx;                /* scatter context for vector */
  PetscBool  roworiented;          /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */
  PetscInt    *rowindices;         /* column indices for row */
  PetscScalar *rowvalues;          /* nonzero values in row */
  PetscBool   getrowactive;        /* indicates MatGetRow(), not restored */

  /* Used by MatDistribute_MPIELL() to allow reuse of previous matrix allocation  and nonzero pattern */
  PetscInt *ld;                    /* number of entries per row left of diagona block */
} Mat_MPIELL;

PETSC_EXTERN PetscErrorCode MatCreate_MPIELL(Mat);
PETSC_INTERN PetscErrorCode MatSetUpMultiply_MPIELL(Mat);

PETSC_INTERN PetscErrorCode MatDisAssemble_MPIELL(Mat);
PETSC_INTERN PetscErrorCode MatDuplicate_MPIELL(Mat,MatDuplicateOption,Mat*);

PETSC_INTERN PetscErrorCode MatDestroy_MPIELL_PtAP(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_MPIELL(Mat);

PETSC_INTERN PetscErrorCode MatSetValues_MPIELL(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar [],InsertMode);
PETSC_INTERN PetscErrorCode MatDestroy_MPIELL_MatMatMult(Mat);
PETSC_INTERN PetscErrorCode MatSetOption_MPIELL(Mat,MatOption,PetscBool);
PETSC_INTERN PetscErrorCode MatGetSeqNonzeroStructure_MPIELL(Mat,Mat*);

PETSC_INTERN PetscErrorCode MatSetFromOptions_MPIELL(PetscOptionItems*,Mat);
PETSC_INTERN PetscErrorCode MatMPIELLSetPreallocation_MPIELL(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);

PETSC_INTERN PetscErrorCode MatConvert_MPIELL_MPIAIJ(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIELL(Mat,MatType,MatReuse,Mat*);

PETSC_INTERN PetscErrorCode MatSOR_MPIELL(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);

PETSC_INTERN PetscErrorCode MatCreateColmap_MPIELL_Private(Mat);
PETSC_INTERN PetscErrorCode MatDiagonalScaleLocal_MPIELL(Mat,Vec);
