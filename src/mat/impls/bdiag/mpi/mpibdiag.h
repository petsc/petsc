/* $Id: mpibdiag.h,v 1.7 1995/10/22 22:23:29 bsmith Exp curfman $ */

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

  int           assembled;              /* MatAssemble has been called */
  InsertMode    insertmode;             /* mode for MatSetValues */
  Stash         stash;                  /* stash for non-local elements */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  int           nsends, nrecvs;         /* numbers of sends and receives */
  Scalar        *svalues, *rvalues;     /* sending and receiving data */
  int           rmax;                   /* maximum message length */
  int           *garray;                /* work array */

  /* The following variables are used for matrix-vector products */

  Vec           lvec;                   /* local vector */
  VecScatter    Mvctx;                  /* scatter context for vector */
} Mat_MPIBDiag;

