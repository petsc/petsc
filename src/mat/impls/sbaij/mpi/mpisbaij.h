/* $Id: mpisbaij.h,v 1.3 2001/08/07 03:03:05 balay Exp $ */

#include "src/mat/impls/baij/seq/baij.h"
#include "src/sys/ctable.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"

#if !defined(__MPISBAIJ_H)
#define __MPISBAIJ_H

typedef struct {
  int           *rowners,*cowners;     /* ranges owned by each processor */
  int           *rowners_bs;           /* rowners*bs */
  int           rstart,rend;           /* starting and ending owned rows */
  int           cstart,cend;           /* starting and ending owned columns */
  Mat           A,B;                   /* local submatrices: 
                                        A (diag part) in SeqSBAIJ format,
                                        B (supper off-diag part) in SeqBAIJ format */
  int           size;                  /* size of communicator */
  int           rank;                  /* rank of proc in communicator */ 
  int           bs,bs2;                /* block size, bs2 = bs*bs */
  int           Mbs,Nbs;               /* number block rows/cols in matrix; M/bs */
  int           mbs,nbs;               /* number block rows/cols on processor; m/bs=n/bs */

  /* The following variables are used for matrix assembly */

  PetscTruth    donotstash;             /* if 1, off processor entries dropped */
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  int           nsends,nrecvs;         /* numbers of sends and receives */
  MatScalar     *svalues,*rvalues;     /* sending and receiving data */
  int           rmax;                   /* maximum message length */
#if defined (PETSC_USE_CTABLE)
  PetscTable    colmap;
#else
  int           *colmap;                /* local col number of off-diag col */
#endif
  int           *garray;                /* work array */

  /* The following variable is used by blocked matrix assembly */
  MatScalar     *barray;                /* Block array of size bs2 */

  /* The following variables are used for matrix-vector products */

  Vec           lvec;                    /* local seq vector */
  VecScatter    Mvctx;            /* scatter context for vector */
  /* Vec           slvec0,slvec1;   */        /* parallel vectors */
  /* VecScatter    sMvctx; */
  PetscTruth    roworiented;             /* if true, row-oriented input, default true */

  /* The following variables are for MatGetRow() */

  int           *rowindices;       /* column indices for row */
  PetscScalar   *rowvalues;        /* nonzero values in row */
  PetscTruth    getrowactive;      /* indicates MatGetRow(), not restored */

  /* Some variables to make MatSetValues and others more efficient */
  int           rstart_bs,rend_bs; 
  int           cstart_bs,cend_bs;
  int           *ht;                      /* Hash table to speed up matrix assembly */
  MatScalar     **hd;                     /* Hash table data */
  int           ht_size;
  int           ht_total_ct,ht_insert_ct; /* Hash table statistics */
  PetscTruth    ht_flag;                  /* Flag to indicate if hash tables are used */
  double        ht_fact;                  /* Factor to determine the HT size */

#if defined(PETSC_USE_MAT_SINGLE)
  int           setvalueslen;
  MatScalar     *setvaluescopy; /* area double precision values in MatSetValuesXXX() are copied
                                      before calling MatSetValuesXXX_MPISBAIJ_MatScalar() */
#endif
} Mat_MPISBAIJ;

#endif
