
#include "aij.h"
#include <math.h>

typedef  struct {int nmax, n, *idx, *idy; Scalar *array;} Stash;

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


