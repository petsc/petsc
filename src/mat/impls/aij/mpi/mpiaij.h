/* $Id: mpiaij.h,v 1.23 2001/08/07 03:02:49 balay Exp $ */

#if !defined(__MPIAIJ_H)
#define __MPIAIJ_H

#include "src/mat/impls/aij/seq/aij.h"
#include "src/sys/ctable.h"

typedef struct {
  int           *rowners,*cowners;     /* ranges owned by each processor */
  int           rstart,rend;           /* starting and ending owned rows */
  int           cstart,cend;           /* starting and ending owned columns */
  Mat           A,B;                   /* local submatrices: A (diag part),
                                           B (off-diag part) */
  int           size;                   /* size of communicator */
  int           rank;                   /* rank of proc in communicator */ 

  /* The following variables are used for matrix assembly */

  PetscTruth    donotstash;             /* PETSC_TRUE if off processor entries dropped */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  int           nsends,nrecvs;         /* numbers of sends and receives */
  PetscScalar   *svalues,*rvalues;     /* sending and receiving data */
  int           rmax;                   /* maximum message length */
#if defined (PETSC_USE_CTABLE)
  PetscTable    colmap;
#else
  int           *colmap;                /* local col number of off-diag col */
#endif
  int           *garray;                /* global index of all off-processor columns */

  /* The following variables are used for matrix-vector products */

  Vec           lvec;              /* local vector */
  VecScatter    Mvctx;             /* scatter context for vector */
  PetscTruth    roworiented;       /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */

  int           *rowindices;       /* column indices for row */
  PetscScalar   *rowvalues;        /* nonzero values in row */
  PetscTruth    getrowactive;      /* indicates MatGetRow(), not restored */

} Mat_MPIAIJ;

EXTERN int MatSetColoring_MPIAIJ(Mat,ISColoring);
EXTERN int MatSetValuesAdic_MPIAIJ(Mat,void*);
EXTERN int MatSetValuesAdifor_MPIAIJ(Mat,int,void*);


#endif
