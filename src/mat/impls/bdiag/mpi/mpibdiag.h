/* $Id: mpibdiag.h,v 1.5 1995/06/21 05:29:05 curfman Exp curfman $ */

#include "bdiag.h"

/* 
   Mat_MPIBDiag - Parallel, block-diagonal format, where each diagonal
   element consists of a square block of size nb x nb.  Dense storage
   within each block is in column-major order.

   For now, the parallel part is just a copy of the Mat_MPIAIJ
   parallel data structure. 
 */

typedef struct {
  int           *rowners,*cowners;  /* ranges owned by each processor */
  int           m,n,M,N;            /* local rows, cols; global rows, cols */
  int           rstart,rend;        /* starting and ending local rows */
  int           brstart,brend;      /* block starting and ending local rows */
  Mat           A;                  /* local matrix */
  int           gnd;                /* number of global diagonals */
  int           *gdiag;             /* global matrix diagonal numbers */
  int           size,rank;
/*  Used in Matrix assembly */
  int           assembled;          /* MatAssemble has been called */
  InsertMode    insertmode;
  Stash         stash;
  MPI_Request   *send_waits,*recv_waits;
  int           nsends,nrecvs;
  Scalar        *svalues,*rvalues;
  int           rmax;
  int           *garray;
/*  Used in Matrix-vector product */
  Vec           lvec;
  VecScatterCtx Mvctx;
} Mat_MPIBDiag;

