
#include "src/mat/impls/bdiag/seq/bdiag.h"

/* 
   Mat_MPIBDiag - Parallel, block-diagonal format, where each diagonal
   element consists of a square block of size bs x bs.  Dense storage
   within each block is in column-major order.

   For now, the parallel part is just a copy of the Mat_MPIAIJ
   parallel data structure. 
 */

typedef struct {
  PetscInt      brstart,brend;      /* block starting and ending local rows */
  Mat           A;                  /* local matrix */
  PetscInt      gnd;                /* number of global diagonals */
  PetscInt      *gdiag;             /* global matrix diagonal numbers */
  PetscMPIInt   size;               /* size of communicator */
  PetscMPIInt   rank;               /* rank of proc in communicator */ 

  /* The following variables are used for matrix assembly */
  PetscTruth    donotstash;             /* 1 if off processor entries dropped */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  PetscInt      nsends,nrecvs;         /* numbers of sends and receives */
  PetscScalar   *svalues,*rvalues;     /* sending and receiving data */
  PetscInt      rmax;                   /* maximum message length */
  PetscInt      *garray;                /* work array */
  PetscTruth    roworiented;            /* indicates MatSetValues() input default 1*/

  /* The following variables are used for matrix-vector products */

  Vec           lvec;                   /* local vector */
  VecScatter    Mvctx;                  /* scatter context for vector */
} Mat_MPIBDiag;

EXTERN PetscErrorCode MatLoad_MPIBDiag(PetscViewer, MatType,Mat*);
EXTERN PetscErrorCode MatSetUpMultiply_MPIBDiag(Mat);
