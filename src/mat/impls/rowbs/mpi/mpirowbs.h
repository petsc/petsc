/* $Id: mpirowbs.h,v 1.31 1996/08/08 14:43:05 bsmith Exp bsmith $ */

#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX) && \
    !defined(__MPIROWBS_H)
#define __MPIROWBS_H

#include "src/mat/matimpl.h"
#include <math.h>

#if defined(__cplusplus)
extern "C" {
#include "BSsparse.h"
#include "BSprivate.h"
}
#else
#include "BSsparse.h"
#include "BSprivate.h"
#endif

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

  InsertMode  insertmode;         /* mode for MatSetValues */
  Stash       stash;              /* stash for non-local elements */
  MPI_Request *send_waits;        /* array of send requests */
  MPI_Request *recv_waits;        /* array of receive requests */
  int         nsends, nrecvs;     /* numbers of sends and receives */
  Scalar      *svalues, *rvalues; /* sending and receiving data */
  int         rmax;               /* maximum message length */
  int         vecs_permscale;     /* flag indicating permuted and scaled vectors */
  int         factor;
  int         mat_is_symmetric;   /* matrix is symmetric; hence use ICC */

  /* BlockSolve data */
  BSprocinfo *procinfo;         /* BlockSolve processor context */
  BSmapping  *bsmap;            /* BlockSolve mapping context */
  BSspmat    *A;                /* initial matrix */
  BSpar_mat  *pA;               /* permuted matrix */
  BScomm     *comm_pA;          /* communication info for triangular solves */
  BSpar_mat  *fpA;              /* factored permuted matrix */
  BScomm     *comm_fpA;         /* communication info for factorization */
  Vec        diag;              /* scaling vector (stores inverse of square
                                   root of permuted diagonal of original matrix) */
  Vec        xwork;             /* work space for mat-vec mult */

  /* Cholesky factorization data */
  double     alpha;             /* restart for failed factorization */
  int        ierr;              /* BS factorization error */
  int        failures;          /* number of BS factorization failures */

  int        mat_is_structurally_symmetric; 

  int        blocksolveassembly;/* Indicates the matrix has been assembled 
                                   for block solve */
} Mat_MPIRowbs;


extern int MatAssemblyEnd_MPIRowbs_ForBlockSolve(Mat);

#define CHKERRBS(a) {if (__BSERROR_STATUS) {fprintf(stderr, \
        "BlockSolve95 Error Code %d\n",__BSERROR_STATUS); CHKERRQ(1);}}

#if defined(PETSC_LOG)  /* turn on BlockSolve logging */
#define MAINLOG
#endif

#endif
