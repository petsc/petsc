/* $Id: mpibaij.h,v 1.4 1996/11/19 16:31:35 bsmith Exp balay $ */

#include "src/mat/impls/baij/seq/baij.h"

#if !defined(__MPIBAIJ_H)
#define __MPIBAIJ_H

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
  int           bs, bs2;                /* block size, bs2 = bs*bs */
  int           Mbs, Nbs;
  int           mbs, nbs;

  /* The following variables are used for matrix assembly */

  InsertMode    insertmode;             /* mode for MatSetValues */
  Stash         stash;                  /* stash for non-local elements */
  int           donotstash;             /* if 1, off processor entries dropped */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  int           nsends, nrecvs;         /* numbers of sends and receives */
  Scalar        *svalues, *rvalues;     /* sending and receiving data */
  int           rmax;                   /* maximum message length */
  int           *colmap;                /* local col number of off-diag col */
  int           *garray;                /* work array */

  /* The following variables are used for matrix-vector products */

  Vec           lvec;              /* local vector */
  VecScatter    Mvctx;             /* scatter context for vector */
  int           roworiented;       /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */

  int           *rowindices;       /* column indices for row */
  Scalar        *rowvalues;        /* nonzero values in row */
  PetscTruth    getrowactive;      /* indicates MatGetRow(), not restored */

  /* Some variables to make MatSetValues and others more efficient */
  int           rstart_bs, rend_bs; 
  int           cstart_bs, cend_bs;
} Mat_MPIBAIJ;


#endif
