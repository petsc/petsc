
#include "src/mat/impls/dense/seq/dense.h"

  /*  Data stuctures for basic parallel dense matrix  */

/* Structure to hold the information for factorization of a dense matrix */
/* Most of this info is used in the pipe send/recv routines */
typedef struct {
  PetscInt    nlnr;        /* number of local rows downstream */
  PetscInt    nrend;       /* rend for downstream processor */
  PetscInt    nbr,pnbr;   /* Down and upstream neighbors */
  PetscInt    *tag;        /* message tags */
  PetscInt    currow;      /* current row number */
  PetscInt    phase;       /* phase (used to indicate tag) */
  PetscInt    up;          /* Are we moving up or down in row number? */
  PetscInt    use_bcast;   /* Are we broadcasting max length? */
  PetscInt    nsend;       /* number of sends */
  PetscInt    nrecv;       /* number of receives */

  /* data initially in matrix context */
  PetscInt    k;           /* Blocking factor (unused as yet) */
  PetscInt    k2;          /* Blocking factor for solves */
  PetscScalar *temp;
  PetscInt    nlptr;
  PetscInt    *lrows;
  PetscInt    *nlrows;
  PetscInt    *pivots;
} FactorCtx;

#define PIPEPHASE (ctx->phase == 0)

typedef struct {
  PetscInt           *rowners,*cowners;     /* ranges owned by each processor */
                                        /* note n == N */
  PetscInt           nvec;                   /* this is the n size for the vector one multiplies with */
  PetscInt           rstart,rend;           /* starting and ending owned rows */
  Mat           A;                      /* local submatrix */
  PetscMPIInt   size;                   /* size of communicator */
  PetscMPIInt   rank;                   /* rank of proc in communicator */ 
  /* The following variables are used for matrix assembly */
  PetscTruth    donotstash;             /* Flag indicationg if values should be stashed */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  PetscInt           nsends,nrecvs;         /* numbers of sends and receives */
  PetscScalar   *svalues,*rvalues;     /* sending and receiving data */
  PetscInt           rmax;                   /* maximum message length */

  /* The following variables are used for matrix-vector products */

  Vec           lvec;                   /* local vector */
  VecScatter    Mvctx;                  /* scatter context for vector */

  PetscTruth    roworiented;            /* if true, row oriented input (default) */
  FactorCtx     *factor;                /* factorization context */
} Mat_MPIDense;

EXTERN PetscErrorCode MatLoad_MPIDense(PetscViewer, MatType,Mat*);
EXTERN PetscErrorCode MatSetUpMultiply_MPIDense(Mat);
EXTERN PetscErrorCode MatGetSubMatrices_MPIDense(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
