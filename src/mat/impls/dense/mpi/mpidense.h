/* $Id: mpidense.h,v 1.4 1995/11/03 02:49:37 bsmith Exp curfman $ */

#include "dense.h"

  /*  Data stuctures for basic parallel dense matrix  */

/* Structure to hold the information for factorization of a dense matrix */
typedef struct {
  int    nlnr;        /* number of local rows downstream */
  int    nbr, pnbr;   /* Down and upstream neighbors */
  int    tag;
  int    k;           /* Blocking factor (unused as yet) */
  int    k2;          /* Blocking factor for solves */
  int    use_bcast;
  double *temp;
} FactCtx;

#define PIPEPHASE (ctx->phase == 0)
/* This stucture is used in the pipe send/recv routines */
typedef struct {
  int nbr, pnbr;
  int nlnr, nlptr, *nlrows;
  int currow; 
  int up;             /* Are we moving up or down in row number? */
  int tag;
  int phase;
  int use_bcast;
  int nsend;
  int nrecv;
} PSPPipe;

typedef struct {
  int           *rowners, *cowners;     /* ranges owned by each processor */
  int           m, n;                   /* local rows and columns */
  int           M, N;                   /* global rows and columns */
  int           rstart, rend;           /* starting and ending owned rows */
  Mat           A;                      /* local submatrix */
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

  /* The following variables are used for matrix-vector products */

  Vec           lvec;                   /* local vector */
  VecScatter    Mvctx;                  /* scatter context for vector */

  FactCtx       *factor;                /* factorization context */
} Mat_MPIDense;
