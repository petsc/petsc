
#include <../src/mat/impls/dense/seq/dense.h>

/*  Data stuctures for basic parallel dense matrix  */

typedef struct { /* used by MatMatMultxxx_MPIDense_MPIDense() */
  Mat            Ae,Be,Ce;           /* matrix in Elemental format */
  PetscErrorCode (*destroy)(Mat);
} Mat_MatMultDense;

typedef struct { /* used by MatTransposeMatMultXXX_MPIDense_MPIDense() */
  PetscScalar    *sendbuf,*atbarray;
  PetscMPIInt    *recvcounts;
  PetscErrorCode (*destroy)(Mat);
  PetscMPIInt    tag;
} Mat_TransMatMultDense;

typedef struct { /* used by MatMatTransposeMultxxx_MPIDense_MPIDense() */
  PetscScalar    *buf[2];
  PetscMPIInt    tag;
  PetscMPIInt    *recvcounts;
  PetscMPIInt    *recvdispls;
  PetscErrorCode (*destroy)(Mat);
  PetscInt       alg; /* algorithm used */
} Mat_MatTransMultDense;

typedef struct {
  PetscInt    nvec;                     /* this is the n size for the vector one multiplies with */
  Mat         A;                        /* local submatrix */
  PetscMPIInt size;                     /* size of communicator */
  PetscMPIInt rank;                     /* rank of proc in communicator */

  /* The following variables are used for matrix assembly */
  PetscBool   donotstash;               /* Flag indicationg if values should be stashed */
  MPI_Request *send_waits;              /* array of send requests */
  MPI_Request *recv_waits;              /* array of receive requests */
  PetscInt    nsends,nrecvs;           /* numbers of sends and receives */
  PetscScalar *svalues,*rvalues;       /* sending and receiving data */
  PetscInt    rmax;                     /* maximum message length */

  /* The following variables are used for matrix-vector products */
  Vec        lvec;                      /* local vector */
  VecScatter Mvctx;                     /* scatter context for vector */
  PetscBool  roworiented;               /* if true, row oriented input (default) */

  Mat_MatTransMatMult   *atb;           /* used by MatTransposeMatMultxxx_MPIAIJ_MPIDense */
  Mat_TransMatMultDense *atbdense;      /* used by MatTransposeMatMultxxx_MPIDense_MPIDense */
  Mat_MatMultDense      *abdense;       /* used by MatMatMultxxx_MPIDense_MPIDense */
  Mat_MatTransMultDense *abtdense;      /* used by MatMatTransposeMultxxx_MPIDense_MPIDense */
} Mat_MPIDense;

PETSC_INTERN PetscErrorCode MatLoad_MPIDense(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatSetUpMultiply_MPIDense(Mat);
PETSC_INTERN PetscErrorCode MatCreateSubMatrices_MPIDense(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
PETSC_INTERN PetscErrorCode MatEqual_MPIDense(Mat,Mat,PetscBool*);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_MPIDense_MPIDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIAIJ_MPIDense(Mat);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIDense(Mat,Mat,Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultSymbolic_MPIDense_MPIDense(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatTransposeMatMultNumeric_MPIDense_MPIDense(Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIDense_MPIAIJ(Mat);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIDense(Mat);

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatMatMultNumeric_MPIDense_MPIDense(Mat,Mat,Mat);
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_Elemental(Mat,Mat,PetscReal,Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_Elemental(Mat,Mat,Mat);
#endif
