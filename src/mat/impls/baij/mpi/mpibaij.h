/* $Id: mpibaij.h,v 1.30 2001/08/07 03:02:58 balay Exp $ */


#if !defined(__MPIBAIJ_H)
#define __MPIBAIJ_H
#include "src/mat/impls/baij/seq/baij.h"
#include "src/sys/ctable.h"

#if defined (PETSC_USE_CTABLE)
#define PETSCTABLE PetscTable;
#else
#define PETSCTABLE int*
#endif

#define MPIBAIJHEADER \
  int           *rowners,*cowners;      /* ranges owned by each processor, in blocks */        \
  int           *rowners_bs;            /* rowners*bs */                                       \
  int           rstart,rend;           /* starting and ending owned rows */                    \
  int           cstart,cend;           /* starting and ending owned columns */                 \
  Mat           A,B;                   /* local submatrices: A (diag part),                    \
                                           B (off-diag part) */                                \
  int           size;                   /* size of communicator */                             \
  int           rank;                   /* rank of proc in communicator */                     \
  int           bs,bs2;                /* block size, bs2 = bs*bs */                           \
  int           Mbs,Nbs;               /* number block rows/cols in matrix; M/bs, N/bs */      \
  int           mbs,nbs;               /* number block rows/cols on processor; m/bs, n/bs */   \
                                                                                               \
  /* The following variables are used for matrix assembly */                                   \
                                                                                               \
  PetscTruth    donotstash;             /* if 1, off processor entries dropped */              \
  MPI_Request   *send_waits;            /* array of send requests */                           \
  MPI_Request   *recv_waits;            /* array of receive requests */                        \
  int           nsends,nrecvs;         /* numbers of sends and receives */                     \
  MatScalar     *svalues,*rvalues;     /* sending and receiving data */                        \
  int           rmax;                   /* maximum message length */                           \
  PETSCTABLE    colmap;                 /* local col number of off-diag col */                 \
                                                                                               \
  int           *garray;                /* work array */                                       \
                                                                                               \
  /* The following variable is used by blocked matrix assembly */                              \
  MatScalar     *barray;                /* Block array of size bs2 */                          \
                                                                                               \
  /* The following variables are used for matrix-vector products */                            \
                                                                                               \
  Vec           lvec;              /* local vector */                                          \
  VecScatter    Mvctx;             /* scatter context for vector */                            \
  PetscTruth    roworiented;       /* if true, row-oriented input, default true */             \
                                                                                               \
  /* The following variables are for MatGetRow() */                                            \
                                                                                               \
  int           *rowindices;       /* column indices for row */                                \
  PetscScalar   *rowvalues;        /* nonzero values in row */                                 \
  PetscTruth    getrowactive;      /* indicates MatGetRow(), not restored */                   \
                                                                                               \
  /* Some variables to make MatSetValues and others more efficient */                          \
  int           rstart_bs,rend_bs;                                                             \
  int           cstart_bs,cend_bs;                                                             \
  int           *ht;                      /* Hash table to speed up matrix assembly */         \
  MatScalar     **hd;                     /* Hash table data */                                \
  int           ht_size;                                                                       \
  int           ht_total_ct,ht_insert_ct; /* Hash table statistics */                          \
  PetscTruth    ht_flag;                  /* Flag to indicate if hash tables are used */       \
  double        ht_fact;                  /* Factor to determine the HT size */                \
                                                                                               \
  int           setvalueslen;    /* only used for single precision computations */             \
  MatScalar     *setvaluescopy; /* area double precision values in MatSetValuesXXX() are copied\
                                      before calling MatSetValuesXXX_MPIBAIJ_MatScalar() */

typedef struct {
  MPIBAIJHEADER
} Mat_MPIBAIJ;

EXTERN int MatLoad_MPIBAIJ(PetscViewer,const MatType,Mat*);
EXTERN int CreateColmap_MPIBAIJ_Private(Mat);
#endif
