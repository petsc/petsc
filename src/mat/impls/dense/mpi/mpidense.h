/* $Id: mpiaij.h,v 1.10 1995/06/07 17:31:43 bsmith Exp curfman $ */

#include "dense.h"

/* for now this is just a copy of Mat_MPIAIJ */

typedef struct {
  int           *rowners, *cowners;     /* ranges owned by each processor */
  int           m, n;                   /* local rows and columns */
  int           M, N;                   /* global rows and columns */
  int           rstart, rend;           /* starting and ending owned rows */
  int           cstart, cend;           /* starting and ending owned columns */
  Mat           A, B;                   /* local submatrices: A (diag part),
                                           B (off-diag part) */
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
  int           *colmap;                /* local col number of off-diag col */
  int           *garray;                /* work array */

  /* The following variables are used for matrix-vector products */

  Vec           lvec;                   /* local vector */
  VecScatterCtx Mvctx;                  /* scatter context for vector */
} Mat_MPIDense;
