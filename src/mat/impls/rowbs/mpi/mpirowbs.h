
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
#include "matimpl.h"
#include <math.h>
#include "bsinterf.h"

#if !defined(__MPIROW_BS_H)
#define __MPIROW_BS_H

/* temporary redefinitions to avoid problems with BlockSolve */
#define PMALLOC(a)       (*PetscMalloc)(a,__LINE__,__FILE__)
#define PFREE(a)         (*PetscFree)(a,__LINE__,__FILE__)
#define PETSCFREE(a)     PFREE(a)
#define PNEW(a)          (a *) PMALLOC(sizeof(a))
#define PMEMCPY(a,b,n)   memcpy((char*)(a),(char*)(b),n)
#define PMEMSET(a,b,n)   memset((char*)(a),(int)(b),n)
#define PSETERR(n,s)     {return PetscError(__LINE__,__DIR__,__FILE__,s,n);}
#define PCHKERR(n)       {if (n) PSETERR(n,(char *)0);}
#define PCHKPTR(a)       if (!a) PSETERR(1,"No memory"); 

/* 
   Mat_MPIRowbs - Parallel, compressed row storage format that's the
   interface to BlockSolve.
 */

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
  Scalar     *inv_diag;

  /* Cholesky factorization data */
  double     alpha;             /* restart for failed factorization */
  int        ierr;              /* BS factorization error */
  int        failures;          /* number of BS factorization failures */
} Mat_MPIRowbs;

/* Add routine declarations that for some strange reason are absent
  in the BS include files */
  void BSforward1();
  void BSbackward1();
  void BSiperm_dvec();
  void BSfor_solve1();
  void BSback_solve1();

#endif
#endif
