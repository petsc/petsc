/* $Id: pdvec.c,v 1.10 1995/06/07 17:30:43 bsmith Exp $ */

#include "aij.h"

typedef struct {
  int           *rowners,*cowners;  /* ranges owned by each processor */
  int           m,n,M,N;            /* local rows, cols, global rows, cols */
  int           rstart,rend,cstart,cend;
  Mat           A,B;                
  int           numtids,mytid;
/*  Used in Matrix assembly */
  int           assembled;          /* MatAssemble has been called */
  InsertMode    insertmode;
  Stash         stash;
  MPI_Request   *send_waits,*recv_waits;
  int           nsends,nrecvs;
  Scalar        *svalues,*rvalues;
  int           rmax;
  int           *colmap;    /* indicates local col number of off proc column*/
  int           *garray;
/*  Used in Matrix-vector product */
  Vec           lvec;
  VecScatterCtx Mvctx;
} Mat_MPIAIJ;
