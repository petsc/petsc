/* $Id: mpirowbs.h,v 1.20 1995/09/12 03:25:35 bsmith Exp bsmith $ */

#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
#include "matimpl.h"
#include <math.h>
#include "BSsparse.h"

#if !defined(__MPIROWBS_H)
#define __MPIROWBS_H

/*
   Mat_MPIRowbs - Parallel, compressed row storage format that's the
   interface to BlockSolve.
 */

typedef struct {
  int           *rowners;           /* range of rows owned by each processor */
  int           m,n,M,N;            /* local rows, cols; global rows, cols */
  int           rstart,rend;        /* starting and ending owned rows */
  int           numtids,mytid;      /* number of procs, my proc ID */
  int           singlemalloc, sorted, roworiented, nonew;
  int           nz, maxnz;          /* total nonzeros stored, allocated */
  int           *imax;              /* allocated matrix space per row */

  /*  Used in Matrix assembly */
  int           assembled;          /* MatAssemble has been called */
  int           reassemble_begun;   /* We're re-assembling */
  InsertMode    insertmode;
  Stash         stash;
  MPI_Request   *send_waits,*recv_waits;
  int           nsends,nrecvs;
  Scalar        *svalues,*rvalues;
  int           rmax;
  int           vecs_permscale;     /* flag indicating permuted and scaled
                                       vectors */
  int           fact_clone;
  int           mat_is_symmetric;   /* indicates matrix is symmetric hence use ICC */

  /* BlockSolve data */
  BSprocinfo *procinfo;         /* BlockSolve processor context */
  BSmapping  *bsmap;            /* BlockSolve mapping context */
  BSspmat    *A;                /* initial matrix */
  BSpar_mat  *pA;               /* permuted matrix */
  BScomm     *comm_pA;          /* communication info for triangular solves */
  BSpar_mat  *fpA;              /* factored permuted matrix */
  BScomm     *comm_fpA;         /* communication info for factorization */
  Vec diag;                     /* diagonal scaling vector */
  Vec xwork;                    /* work space for mat-vec mult */
  Scalar     *inv_diag;

  /* Cholesky factorization data */
  double     alpha;             /* restart for failed factorization */
  int        ierr;              /* BS factorization error */
  int        failures;          /* number of BS factorization failures */

  int        mat_is_structurally_symmetric; 
} Mat_MPIRowbs;

#define CHKERRBS(a) {if (__BSERROR_STATUS) {fprintf(stderr, \
        "BlockSolve Error Code %d\n",__BSERROR_STATUS); CHKERRQ(a);}}

#endif
#endif
