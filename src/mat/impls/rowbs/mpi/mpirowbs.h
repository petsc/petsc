/* $Id: mpirowbs.h,v 1.23 1995/12/02 19:22:51 curfman Exp bsmith $ */

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
  int         *rowners;           /* range of rows owned by each proc */
  int         m, n;               /* local rows and columns */
  int         M, N;               /* global rows and columns */
  int         rstart, rend;       /* starting and ending owned rows */
  int         size;               /* size of communicator */
  int         rank;               /* rank of proc in communicator */ 
  int         sorted;             /* if true, rows sorted by increasing cols */
  int         roworiented;        /* if true, row-oriented storage */
  int         nonew;              /* if true, no new elements allowed */
  int         nz, maxnz;          /* total nonzeros stored, allocated */
  int         *imax;              /* allocated matrix space per row */

  /*  The following variables are used in matrix assembly */

  int         assembled;          /* MatAssemble has been called */
  int         reassemble_begun;   /* We're re-assembling */
  InsertMode  insertmode;         /* mode for MatSetValues */
  Stash       stash;              /* stash for non-local elements */
  MPI_Request *send_waits;        /* array of send requests */
  MPI_Request *recv_waits;        /* array of receive requests */
  int         nsends, nrecvs;     /* numbers of sends and receives */
  Scalar      *svalues, *rvalues; /* sending and receiving data */
  int         rmax;               /* maximum message length */
  int         vecs_permscale;     /* flag indicating permuted and scaled vectors */
  int         fact_clone;
  int         mat_is_symmetric;   /* matrix is symmetric; hence use ICC */

  /* BlockSolve data */
  BSprocinfo *procinfo;         /* BlockSolve processor context */
  BSmapping  *bsmap;            /* BlockSolve mapping context */
  BSspmat    *A;                /* initial matrix */
  BSpar_mat  *pA;               /* permuted matrix */
  BScomm     *comm_pA;          /* communication info for triangular solves */
  BSpar_mat  *fpA;              /* factored permuted matrix */
  BScomm     *comm_fpA;         /* communication info for factorization */
  Vec        diag;              /* diagonal scaling vector */
  Vec        xwork;             /* work space for mat-vec mult */
  Scalar     *inv_diag;

  /* Cholesky factorization data */
  double     alpha;             /* restart for failed factorization */
  int        ierr;              /* BS factorization error */
  int        failures;          /* number of BS factorization failures */

  int        mat_is_structurally_symmetric; 
} Mat_MPIRowbs;

#define CHKERRBS(a) {if (__BSERROR_STATUS) {fprintf(stderr, \
        "BlockSolve Error Code %d\n",__BSERROR_STATUS); CHKERRQ(a);}}

#if defined(PETSC_LOG)  /* turn on BlockSolve logging */
#define MAINLOG
#endif

#endif
#endif
