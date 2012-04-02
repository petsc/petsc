
#if !defined(__MPIAIJ_H)
#define __MPIAIJ_H

#include <../src/mat/impls/aij/seq/aij.h>

typedef struct { /* used by MatCreateMPIAIJSumSeqAIJ for reusing the merged matrix */
  PetscLayout    rowmap;
  PetscInt       **buf_ri,**buf_rj;
  PetscMPIInt    *len_s,*len_r,*id_r; /* array of length of comm->size, store send/recv matrix values */
  PetscMPIInt    nsend,nrecv;  
  PetscInt       *bi,*bj; /* i and j array of the local portion of mpi C (matrix product) - rename to ci, cj! */
  PetscInt       *owners_co,*coi,*coj; /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*duplicate)(Mat,MatDuplicateOption,Mat*);
} Mat_Merge_SeqsToMPI; 

typedef struct { /* used by MatPtAP_MPIAIJ_MPIAIJ() and MatMatMult_MPIAIJ_MPIAIJ() */
  PetscInt       *startsj_s,*startsj_r; /* used by MatGetBrowsOfAoCols_MPIAIJ */
  PetscScalar    *bufa;                 /* used by MatGetBrowsOfAoCols_MPIAIJ */
  Mat            P_loc,P_oth;  /* partial B_seq -- intend to replace B_seq */
  PetscInt       *api,*apj;    /* symbolic i and j arrays of the local product A_loc*B_seq */
  PetscInt       rmax;         /* max num of nnz in a local row of the matrix product */
  MatReuse       reuse;        /* flag to skip MatGetBrowsOfAoCols_MPIAIJ() and MatMPIAIJGetLocalMat() in 1st call of MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  PetscScalar    *apa;         /* tmp array for store a row of A*P used in MatMatMult() */
  Mat            A_loc;        /* used by MatTransposeMatMult(), contains api and apj */

  Mat_Merge_SeqsToMPI *merge;
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*duplicate)(Mat,MatDuplicateOption,Mat*);
} Mat_PtAPMPI;

typedef struct {
  Mat           A,B;                   /* local submatrices: A (diag part),
                                           B (off-diag part) */
  PetscMPIInt   size;                   /* size of communicator */
  PetscMPIInt   rank;                   /* rank of proc in communicator */ 

  /* The following variables are used for matrix assembly */
  PetscBool     donotstash;             /* PETSC_TRUE if off processor entries dropped */
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
  PetscBool     roworiented;       /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */
  PetscInt      *rowindices;       /* column indices for row */
  PetscScalar   *rowvalues;        /* nonzero values in row */
  PetscBool     getrowactive;      /* indicates MatGetRow(), not restored */

  /* Used by MatDistribute_MPIAIJ() to allow reuse of previous matrix allocation  and nonzero pattern */
  PetscInt      *ld;               /* number of entries per row left of diagona block */

  /* Used by MatMatMult() and MatPtAP() */
  Mat_PtAPMPI   *ptap;
} Mat_MPIAIJ;

extern PetscErrorCode MatSetColoring_MPIAIJ(Mat,ISColoring);
extern PetscErrorCode MatSetValuesAdic_MPIAIJ(Mat,void*);
extern PetscErrorCode MatSetValuesAdifor_MPIAIJ(Mat,PetscInt,void*);
extern PetscErrorCode MatSetUpMultiply_MPIAIJ(Mat);
extern PetscErrorCode MatDisAssemble_MPIAIJ(Mat);
extern PetscErrorCode MatDuplicate_MPIAIJ(Mat,MatDuplicateOption,Mat *);
extern PetscErrorCode MatIncreaseOverlap_MPIAIJ(Mat,PetscInt,IS [],PetscInt);
extern PetscErrorCode MatFDColoringCreate_MPIAIJ(Mat,ISColoring,MatFDColoring);
extern PetscErrorCode MatGetSubMatrices_MPIAIJ (Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
extern PetscErrorCode MatGetSubMatrix_MPIAIJ_All(Mat,MatGetSubMatrixOption,MatReuse,Mat *[]);
extern PetscErrorCode MatGetSubMatricesParallel_MPIAIJ (Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);

extern PetscErrorCode MatGetSubMatrix_MPIAIJ(Mat,IS,IS,MatReuse,Mat *);
extern PetscErrorCode MatGetSubMatrix_MPIAIJ_Private (Mat,IS,IS,PetscInt,MatReuse,Mat *);
extern PetscErrorCode MatGetMultiProcBlock_MPIAIJ(Mat,MPI_Comm,MatReuse,Mat*);

extern PetscErrorCode MatLoad_MPIAIJ(Mat,PetscViewer);
extern PetscErrorCode MatMatMult_MPIDense_MPIAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode MatMatMult_MPIAIJ_MPIAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat,Mat,Mat);

extern PetscErrorCode MatPtAPSymbolic_MPIAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatPtAPNumeric_MPIAIJ(Mat,Mat,Mat);
extern PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat,Mat,Mat);
extern PetscErrorCode MatDestroy_MPIAIJ_PtAP(Mat);

extern PetscErrorCode MatGetBrowsOfAoCols_MPIAIJ(Mat,Mat,MatReuse,PetscInt**,PetscInt**,MatScalar**,Mat*);
extern PetscErrorCode MatSetValues_MPIAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar [],InsertMode);
extern PetscErrorCode MatDestroy_MPIAIJ_MatMatMult(Mat);
extern PetscErrorCode PetscContainerDestroy_Mat_MatMatMultMPI(void*);

extern PetscErrorCode MatTransposeMatMult_MPIAIJ_MPIAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatTransposeMatMultSymbolic_MPIAIJ_MPIAIJ_Scalable(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ(Mat,Mat,Mat);

extern PetscErrorCode MatGetRedundantMatrix_MPIAIJ(Mat,PetscInt,MPI_Comm,PetscInt,MatReuse,Mat*);
extern PetscErrorCode MatGetSeqNonzeroStructure_MPIAIJ(Mat,Mat*);

EXTERN_C_BEGIN
extern PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN_C_END

#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_REAL_SINGLE) && !defined(PETSC_USE_REAL___FLOAT128)
extern PetscErrorCode MatLUFactorSymbolic_MPIAIJ_TFS(Mat,IS,IS,const MatFactorInfo*,Mat*);
#endif
extern PetscErrorCode MatSolve_MPIAIJ(Mat,Vec,Vec);
extern PetscErrorCode MatILUFactor_MPIAIJ(Mat,IS,IS,const MatFactorInfo *);

EXTERN_C_BEGIN
extern PetscErrorCode  MatGetDiagonalBlock_MPIAIJ(Mat,Mat *);
extern PetscErrorCode  MatDiagonalScaleLocal_MPIAIJ(Mat,Vec);
EXTERN_C_END

#endif
