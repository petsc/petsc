
#if !defined(__MPIAIJ_H)
#define __MPIAIJ_H

#include "src/mat/impls/aij/seq/aij.h"
#include "src/sys/ctable.h"

typedef struct {
  PetscInt      *rowners,*cowners;     /* ranges owned by each processor */
  PetscInt      rstart,rend;           /* starting and ending owned rows */
  PetscInt      cstart,cend;           /* starting and ending owned columns */
  Mat           A,B;                   /* local submatrices: A (diag part),
                                           B (off-diag part) */
  PetscMPIInt   size;                   /* size of communicator */
  PetscMPIInt   rank;                   /* rank of proc in communicator */ 

  /* The following variables are used for matrix assembly */

  PetscTruth    donotstash;             /* PETSC_TRUE if off processor entries dropped */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  PetscInt      nsends,nrecvs;         /* numbers of sends and receives */
  PetscScalar   *svalues,*rvalues;     /* sending and receiving data */
  PetscInt      rmax;                   /* maximum message length */
#if defined (PETSC_USE_CTABLE)
  PetscTable    colmap;
#else
  PetscInt      *colmap;                /* local col number of off-diag col */
#endif
  PetscInt      *garray;                /* global index of all off-processor columns */

  /* The following variables are used for matrix-vector products */

  Vec           lvec;              /* local vector */
  VecScatter    Mvctx;             /* scatter context for vector */
  PetscTruth    roworiented;       /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */

  PetscMPIInt   *rowindices;       /* column indices for row */
  PetscScalar   *rowvalues;        /* nonzero values in row */
  PetscTruth    getrowactive;      /* indicates MatGetRow(), not restored */

} Mat_MPIAIJ;

typedef struct { /* used by MatMatMult_MPIAIJ_MPIAIJ for reusing symbolic mat product */
  IS       isrowa,isrowb,iscolb;
  Mat      *aseq,*bseq,C_seq; /* A_seq=aseq[0], B_seq=bseq[0] */
  PetscInt brstart; /* starting owned rows of B in matrix bseq[0]; brend = brstart+B->m */
} Mat_MatMatMultMPI;

EXTERN PetscErrorCode MatSetColoring_MPIAIJ(Mat,ISColoring);
EXTERN PetscErrorCode MatSetValuesAdic_MPIAIJ(Mat,void*);
EXTERN PetscErrorCode MatSetValuesAdifor_MPIAIJ(Mat,PetscInt,void*);
EXTERN PetscErrorCode MatSetUpMultiply_MPIAIJ(Mat);
EXTERN PetscErrorCode DisAssemble_MPIAIJ(Mat);
EXTERN PetscErrorCode MatDuplicate_MPIAIJ(Mat,MatDuplicateOption,Mat *);
EXTERN PetscErrorCode MatIncreaseOverlap_MPIAIJ(Mat,PetscInt,IS [],PetscInt);
EXTERN PetscErrorCode MatFDColoringCreate_MPIAIJ(Mat,ISColoring,MatFDColoring);
EXTERN PetscErrorCode MatGetSubMatrices_MPIAIJ (Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
EXTERN PetscErrorCode MatGetSubMatrix_MPIAIJ (Mat,IS,IS,PetscInt,MatReuse,Mat *);
EXTERN PetscErrorCode MatLoad_MPIAIJ(PetscViewer,const MatType,Mat*);
EXTERN PetscErrorCode MatMatMult_MPIAIJ_MPIAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatPtAP_MPIAIJ_MPIAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatSetValues_MPIAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar [],InsertMode);


#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN PetscErrorCode MatLUFactorSymbolic_MPIAIJ_TFS(Mat,IS,IS,MatFactorInfo*,Mat*);
#endif

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatGetDiagonalBlock_MPIAIJ(Mat,PetscTruth *,MatReuse,Mat *);
EXTERN PetscErrorCode MatDiagonalScaleLocal_MPIAIJ(Mat,Vec);
EXTERN_C_END

#endif
