/* $Id: mpibdiag.h,v 1.9 1995/12/30 22:32:49 bsmith Exp bsmith $ */

#include "bdiag.h"

/* 
   Mat_MPIBDiag - Parallel, block-diagonal format, where each diagonal
   element consists of a square block of size nb x nb.  Dense storage
   within each block is in column-major order.

   For now, the parallel part is just a copy of the Mat_MPIAIJ
   parallel data structure. 
 */

typedef struct {
  int           *rowners;           /* row range owned by each processor */
  int           m,n,M,N;            /* local rows, cols; global rows, cols */
  int           rstart,rend;        /* starting and ending local rows */
  int           brstart,brend;      /* block starting and ending local rows */
  Mat           A;                  /* local matrix */
  int           gnd;                /* number of global diagonals */
  int           *gdiag;             /* global matrix diagonal numbers */
  int           size;                   /* size of communicator */
  int           rank;                   /* rank of proc in communicator */ 

  /* The following variables are used for matrix assembly */

  InsertMode    insertmode;             /* mode for MatSetValues */
  Stash         stash;                  /* stash for non-local elements */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  int           nsends, nrecvs;         /* numbers of sends and receives */
  Scalar        *svalues, *rvalues;     /* sending and receiving data */
  int           rmax;                   /* maximum message length */
  int           *garray;                /* work array */
  int           roworiented;            /* indicates MatSetValues() input default 1*/

  /* The following variables are used for matrix-vector products */

  Vec           lvec;                   /* local vector */
  VecScatter    Mvctx;                  /* scatter context for vector */
} Mat_MPIBDiag;

