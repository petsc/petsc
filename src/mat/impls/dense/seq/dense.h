
#if !defined(__DENSE_H)
#define __DENSE_H
#include <petsc/private/matimpl.h>
/* TODO REMOVE */
#include <../src/mat/impls/aij/seq/aij.h> /* Mat_MatTransMatMult is defined here */

/*
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  PetscScalar  *v;                /* matrix elements */
  PetscScalar  *unplacedarray;    /* if one called MatDensePlaceArray(), this is where it stashed the original */
  PetscBool    roworiented;       /* if true, row oriented input (default) */
  PetscInt     pad;               /* padding */
  PetscBLASInt *pivots;           /* pivots in LU factorization */
  PetscBLASInt lfwork;            /* length of work array in factorization */
  PetscScalar  *fwork;            /* work array in factorization */
  PetscScalar  *tau;              /* scalar factors of QR factorization */
  Vec          qrrhs;            /* RHS for solving with QR (solution vector can't hold copy of RHS) */
  PetscBLASInt lda;               /* Lapack leading dimension of data */
  PetscBLASInt rank;              /* numerical rank (of a QR factorized matrix) */
  PetscBool    user_alloc;        /* true if the user provided the dense data */
  PetscBool    unplaced_user_alloc;
  Mat          ptapwork;          /* workspace (SeqDense matrix) for PtAP */

  /* Support for MatDenseGetColumnVec and MatDenseGetSubMatrix */
  Mat               cmat;      /* matrix representation of a given subset of columns */
  Vec               cvec;      /* vector representation of a given column */
  const PetscScalar *ptrinuse; /* holds array to be restored (just a placeholder) */
  PetscInt          vecinuse;  /* if cvec is in use (col = vecinuse-1) */
  PetscInt          matinuse;  /* if cmat is in use (cbegin = matinuse-1) */
} Mat_SeqDense;

PETSC_INTERN PetscErrorCode MatMatMultSymbolic_SeqDense_SeqDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDense_SeqDense(Mat,Mat,Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_SeqDense_SeqDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultNumeric_SeqDense_SeqDense(Mat,Mat,Mat);
PETSC_INTERN PetscErrorCode MatMatTransposeMultSymbolic_SeqDense_SeqDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatMatTransposeMultNumeric_SeqDense_SeqDense(Mat,Mat,Mat);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqDense(Mat,Mat,Mat);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_SeqBAIJ_SeqDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqBAIJ_SeqDense(Mat,Mat,Mat);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_SeqSBAIJ_SeqDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqSBAIJ_SeqDense(Mat,Mat,Mat);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_Nest_Dense(Mat,Mat,PetscReal,Mat*);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_Nest_Dense(Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqDense(Mat);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense(Mat);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqXBAIJ_SeqDense(Mat);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqDense_SeqAIJ(Mat);

PETSC_EXTERN PetscErrorCode MatCreate_SeqDense(Mat);

/* Used by SeqDenseCUDA */
PETSC_INTERN PetscErrorCode MatDuplicateNoCreate_SeqDense(Mat,Mat,MatDuplicateOption);
PETSC_INTERN PetscErrorCode MatNorm_SeqDense(Mat,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode MatView_SeqDense(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatDestroy_SeqDense(Mat);
PETSC_INTERN PetscErrorCode MatDenseGetArray_SeqDense(Mat,PetscScalar*[]);
PETSC_INTERN PetscErrorCode MatDenseRestoreArray_SeqDense(Mat,PetscScalar*[]);
PETSC_INTERN PetscErrorCode MatAXPY_SeqDense(Mat,PetscScalar,Mat,MatStructure);
PETSC_INTERN PetscErrorCode MatMultTransposeAdd_SeqDense(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqDense(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqDense(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqDense(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatDuplicate_SeqDense(Mat,MatDuplicateOption,Mat*);
PETSC_INTERN PetscErrorCode MatSeqDenseSetPreallocation_SeqDense(Mat,PetscScalar*);
PETSC_INTERN PetscErrorCode MatCholeskyFactor_SeqDense(Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactor_SeqDense(Mat,IS,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqDense(Mat,Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatLUFactorSymbolic_SeqDense(Mat,Mat,IS,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSeqDenseSymmetrize_Private(Mat,PetscBool);
PETSC_INTERN PetscErrorCode MatGetColumnVector_SeqDense(Mat,Vec,PetscInt);
PETSC_INTERN PetscErrorCode MatScale_SeqDense(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVec_SeqDense(Mat,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVec_SeqDense(Mat,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVecRead_SeqDense(Mat,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVecRead_SeqDense(Mat,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVecWrite_SeqDense(Mat,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVecWrite_SeqDense(Mat,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode MatDenseGetSubMatrix_SeqDense(Mat,PetscInt,PetscInt,Mat*);
PETSC_INTERN PetscErrorCode MatDenseRestoreSubMatrix_SeqDense(Mat,Mat*);
PETSC_INTERN PetscErrorCode MatDenseSetLDA_SeqDense(Mat,PetscInt);
PETSC_INTERN PetscErrorCode MatCopy_SeqDense(Mat,Mat,MatStructure);

#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode MatSeqDenseCUDAInvertFactors_Private(Mat);
PETSC_INTERN PetscErrorCode MatConvert_SeqDenseCUDA_SeqDense(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqDenseCUDA(Mat,MatType,MatReuse,Mat*);
#endif

PETSC_EXTERN PetscErrorCode MatSeqDenseInvertFactors_Private(Mat);

PETSC_INTERN PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqDense(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPIDense(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);

PETSC_INTERN PetscErrorCode MatView_Dense_Binary(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatLoad_Dense_Binary(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatLoad_Dense_HDF5(Mat,PetscViewer);
#endif
