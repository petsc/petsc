
#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
#include "matimpl.h"
#include <math.h>
#include "bsinterf.h"

#if !defined(__MPIROW_BS_H)
#define __MPIROW_BS_H

/* 
   Mat_MPIRowbs - Parallel, compressed row storage format that's the
   interface to BlockSolve.
 */

/* For now, this is just the same as the MPIAIJ stash ... */
typedef  struct {int nmax, n, *idx, *idy; Scalar *array;} Stash3;

typedef struct {
  int           *rowners;           /* range of rows owned by each processor */
  int           m,n,M,N;            /* local rows, cols, global rows, cols */
  int           rstart,rend;
  int           numtids,mytid;
  int           singlemalloc, sorted, roworiented, nonew;
  int           nz, maxnz;          /* total nonzeros stored, allocated */
  int           mem;                /* total memory */
  int           *imax;              /* Allocated matrix space per row */

  /*  Used in Matrix assembly */
  int           assembled;          /* MatAssemble has been called */
  InsertMode    insertmode;
  Stash3        stash;
  MPI_Request   *send_waits,*recv_waits;
  int           nsends,nrecvs;
  Scalar        *svalues,*rvalues;
  int           rmax;
  int           ctx_filled;         /* matrix context has been filled */
  int           vecs_permuted;      /* flag indicating permuted vectors */

  /* BlockSolve data */
  BSprocinfo *procinfo;
  BSmapping  *bsmap;
  BSspmat    *A;                /* initial matrix */
  BSpar_mat  *pA;               /* permuted matrix */
  BScomm     *comm_pA;          /* communication info for triangular solves */
  BSpar_mat  *fpA;              /* factored permuted matrix */
  BScomm     *comm_fpA;         /* communication info for factorization */
  Vec diag;                     /* diagonal scaling vector */
  Vec xwork;                    /* work space for mat-vec mult */

  /* Cholesky factorization data */
  double     alpha;             /* restart for failed factorization */
  int        ierr;              /* BS factorization error */
  int        failures;          /* number of BS factorization failures */
} Mat_MPIRowbs;

#endif
#endif


