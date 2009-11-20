
#if !defined(__MPIAIJ_H)
#define __MPIAIJ_H

#include "../src/mat/impls/aij/seq/aij.h"
#include "../src/sys/ctable.h"

typedef struct {
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
  Vec           diag;
  VecScatter    Mvctx;             /* scatter context for vector */
  PetscTruth    roworiented;       /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */

  PetscInt      *rowindices;       /* column indices for row */
  PetscScalar   *rowvalues;        /* nonzero values in row */
  PetscTruth    getrowactive;      /* indicates MatGetRow(), not restored */

  /* Used by MatDistribute_MPIAIJ() to allow reuse of previous matrix allocation  and nonzero pattern */
  PetscInt      *ld;               /* number of entries per row left of diagona block */
} Mat_MPIAIJ;

typedef struct { /* used by MatMatMult_MPIAIJ_MPIAIJ and MatPtAP_MPIAIJ_MPIAIJ for reusing symbolic mat product */
  PetscInt       *startsj,*startsj_r;
  PetscScalar    *bufa;
  IS             isrowa,isrowb,iscolb; 
  Mat            *aseq,*bseq,C_seq; /* A_seq=aseq[0], B_seq=bseq[0] */
  Mat            A_loc,B_seq;
  Mat            B_loc,B_oth;  /* partial B_seq -- intend to replace B_seq */
  PetscInt       brstart; /* starting owned rows of B in matrix bseq[0]; brend = brstart+B->m */
  PetscInt       *abi,*abj; /* symbolic i and j arrays of the local product A_loc*B_seq */
  PetscInt       abnz_max;  /* max(abi[i+1] - abi[i]), max num of nnz in a row of A_loc*B_seq */
  MatReuse       reuse; 
  PetscErrorCode (*MatDestroy)(Mat);
  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
} Mat_MatMatMultMPI;

typedef struct { /* used by MatMerge_SeqsToMPI for reusing the merged matrix */
  PetscLayout    rowmap;
  PetscInt       **buf_ri,**buf_rj;
  PetscMPIInt    *len_s,*len_r,*id_r; /* array of length of comm->size, store send/recv matrix values */
  PetscMPIInt    nsend,nrecv;  
  PetscInt       *bi,*bj; /* i and j array of the local portion of mpi C=P^T*A*P */
  PetscInt       *owners_co,*coi,*coj; /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscErrorCode (*MatDestroy)(Mat);
  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
} Mat_Merge_SeqsToMPI; 

EXTERN PetscErrorCode MatSetColoring_MPIAIJ(Mat,ISColoring);
EXTERN PetscErrorCode MatSetValuesAdic_MPIAIJ(Mat,void*);
EXTERN PetscErrorCode MatSetValuesAdifor_MPIAIJ(Mat,PetscInt,void*);
EXTERN PetscErrorCode MatSetUpMultiply_MPIAIJ(Mat);
EXTERN PetscErrorCode DisAssemble_MPIAIJ(Mat);
EXTERN PetscErrorCode MatDuplicate_MPIAIJ(Mat,MatDuplicateOption,Mat *);
EXTERN PetscErrorCode MatIncreaseOverlap_MPIAIJ(Mat,PetscInt,IS [],PetscInt);
EXTERN PetscErrorCode MatFDColoringCreate_MPIAIJ(Mat,ISColoring,MatFDColoring);
EXTERN PetscErrorCode MatGetSubMatrices_MPIAIJ (Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
EXTERN PetscErrorCode MatGetSubMatrix_MPIAIJ_All(Mat,MatGetSubMatrixOption,MatReuse,Mat *[]);

EXTERN PetscErrorCode MatGetSubMatrix_MPIAIJ(Mat,IS,IS,MatReuse,Mat *);
EXTERN PetscErrorCode MatGetSubMatrix_MPIAIJ_Private (Mat,IS,IS,PetscInt,MatReuse,Mat *);
EXTERN PetscErrorCode MatLoad_MPIAIJ(PetscViewer, const MatType,Mat*);
EXTERN PetscErrorCode MatMatMult_MPIAIJ_MPIAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatPtAPNumeric_MPIAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatSetValues_MPIAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar [],InsertMode);
EXTERN PetscErrorCode MatDestroy_MPIAIJ_MatMatMult(Mat);
EXTERN PetscErrorCode PetscContainerDestroy_Mat_MatMatMultMPI(void*);
EXTERN PetscErrorCode MatGetRedundantMatrix_MPIAIJ(Mat,PetscInt,MPI_Comm,PetscInt,MatReuse,Mat*);
EXTERN PetscErrorCode MatGetSeqNonzeroStructure_MPIAIJ(Mat,Mat*);

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN_C_END

#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SCALAR_SINGLE) && !defined(PETSC_USE_SCALAR_MAT_SINGLE)
EXTERN PetscErrorCode MatLUFactorSymbolic_MPIAIJ_TFS(Mat,IS,IS,const MatFactorInfo*,Mat*);
#endif
EXTERN PetscErrorCode MatSolve_MPIAIJ(Mat,Vec,Vec);
EXTERN PetscErrorCode MatILUFactor_MPIAIJ(Mat,IS,IS,const MatFactorInfo *);

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetDiagonalBlock_MPIAIJ(Mat,PetscTruth *,MatReuse,Mat *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatDiagonalScaleLocal_MPIAIJ(Mat,Vec);

EXTERN_C_END

#endif
