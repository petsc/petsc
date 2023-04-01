
#include <../src/mat/impls/dense/seq/dense.h>
#include <petscsf.h>

/*  Data structures for basic parallel dense matrix  */

typedef struct {  /* used by MatMatMultxxx_MPIDense_MPIDense() */
  Mat Ae, Be, Ce; /* matrix in Elemental format */
} Mat_MatMultDense;

typedef struct { /* used by MatTransposeMatMultXXX_MPIDense_MPIDense() */
  PetscScalar *sendbuf;
  Mat          atb;
  PetscMPIInt *recvcounts;
  PetscMPIInt  tag;
} Mat_TransMatMultDense;

typedef struct { /* used by MatMatTransposeMultxxx_MPIDense_MPIDense() */
  PetscScalar *buf[2];
  PetscMPIInt  tag;
  PetscMPIInt *recvcounts;
  PetscMPIInt *recvdispls;
  PetscInt     alg; /* algorithm used */
} Mat_MatTransMultDense;

typedef struct {
  Mat A; /* local submatrix */

  /* The following variables are used for matrix assembly */
  PetscBool    donotstash;        /* Flag indicating if values should be stashed */
  MPI_Request *send_waits;        /* array of send requests */
  MPI_Request *recv_waits;        /* array of receive requests */
  PetscInt     nsends, nrecvs;    /* numbers of sends and receives */
  PetscScalar *svalues, *rvalues; /* sending and receiving data */
  PetscInt     rmax;              /* maximum message length */

  /* The following variables are used for matrix-vector products */
  Vec       lvec;        /* local vector */
  PetscSF   Mvctx;       /* for mat-mult communications */
  PetscBool roworiented; /* if true, row oriented input (default) */

  /* Support for MatDenseGetColumnVec and MatDenseGetSubMatrix */
  Mat                cmat;     /* matrix representation of a given subset of columns */
  Vec                cvec;     /* vector representation of a given column */
  const PetscScalar *ptrinuse; /* holds array to be restored (just a placeholder) */
  PetscInt           vecinuse; /* if cvec is in use (col = vecinuse-1) */
  PetscInt           matinuse; /* if cmat is in use (cbegin = matinuse-1) */
} Mat_MPIDense;

PETSC_INTERN PetscErrorCode MatSetUpMultiply_MPIDense(Mat);
PETSC_INTERN PetscErrorCode MatCreateSubMatrices_MPIDense(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *[]);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIAIJ_MPIDense(Mat);

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIDense_MPIAIJ(Mat);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIDense(Mat);

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_Elemental(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_Elemental(Mat, Mat, Mat);
#endif
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat);

PETSC_INTERN PetscErrorCode MatShift_MPIDense(Mat, PetscScalar);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVecWrite_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVecRead_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVec_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVecWrite_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVecRead_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVec_MPIDense(Mat, PetscInt, Vec *);

PETSC_INTERN PetscErrorCode MatCreate_MPIDense(Mat);

#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode MatConvert_MPIDense_MPIDenseCUDA(Mat, MatType, MatReuse, Mat *);
#endif

#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode MatConvert_MPIDense_MPIDenseHIP(Mat, MatType, MatReuse, Mat *);
#endif
