
#if !defined(__MPIBAIJ_H)
#define __MPIBAIJ_H
#include "../src/mat/impls/baij/seq/baij.h"
#include "../src/sys/ctable.h"

#if defined (PETSC_USE_CTABLE)
#define PETSCTABLE PetscTable
#else
#define PETSCTABLE PetscInt*
#endif

#define MPIBAIJHEADER \
  PetscInt      *rangebs;              /* rmap->range/bs */                      		\
  PetscInt      rstartbs,rendbs,cstartbs,cendbs;  /* map values / bs  */                        \
  Mat           A,B;                   /* local submatrices: A (diag part), B (off-diag part) */ \
  PetscMPIInt   size;                   /* size of communicator */                             \
  PetscMPIInt   rank;                   /* rank of proc in communicator */                     \
  PetscInt      bs2;                    /* block size, bs2 = bs*bs */                           \
  PetscInt      Mbs,Nbs;               /* number block rows/cols in matrix; M/bs, N/bs */      \
  PetscInt      mbs,nbs;               /* number block rows/cols on processor; m/bs, n/bs */   \
                                                                                               \
  /* The following variables are used for matrix assembly */                                   \
                                                                                               \
  PetscTruth    donotstash;             /* if 1, off processor entries dropped */              \
  MPI_Request   *send_waits;            /* array of send requests */                           \
  MPI_Request   *recv_waits;            /* array of receive requests */                        \
  PetscInt      nsends,nrecvs;         /* numbers of sends and receives */                     \
  MatScalar     *svalues,*rvalues;     /* sending and receiving data */                        \
  PetscInt      rmax;                   /* maximum message length */                           \
  PETSCTABLE    colmap;                 /* local col number of off-diag col */                 \
                                                                                               \
  PetscInt     *garray;                /* work array */                                       \
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
  PetscInt      *rowindices;       /* column indices for row */                                \
  PetscScalar   *rowvalues;        /* nonzero values in row */                                 \
  PetscTruth    getrowactive;      /* indicates MatGetRow(), not restored */                   \
                                                                                               \
  /* Some variables to make MatSetValues and others more efficient */                          \
  PetscInt      rstart_bs,rend_bs;                                                             \
  PetscInt      cstart_bs,cend_bs;                                                             \
  PetscInt      *ht;                      /* Hash table to speed up matrix assembly */         \
  MatScalar     **hd;                     /* Hash table data */                                \
  PetscInt      ht_size;                                                                       \
  PetscInt      ht_total_ct,ht_insert_ct; /* Hash table statistics */                          \
  PetscTruth    ht_flag;                  /* Flag to indicate if hash tables are used */       \
  double        ht_fact;                  /* Factor to determine the HT size */                \
                                                                                               \
  PetscInt      setvalueslen;    /* only used for single precision computations */             \
  MatScalar     *setvaluescopy /* area double precision values in MatSetValuesXXX() are copied*/ \
                                   /*   before calling MatSetValuesXXX_MPIBAIJ_MatScalar() */

typedef struct {
  MPIBAIJHEADER;
} Mat_MPIBAIJ;

EXTERN PetscErrorCode MatLoad_MPIBAIJ(PetscViewer, const MatType,Mat*);
EXTERN PetscErrorCode CreateColmap_MPIBAIJ_Private(Mat);
EXTERN PetscErrorCode MatGetSubMatrices_MPIBAIJ(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat*[]);
EXTERN PetscErrorCode MatGetSubMatrix_MPIBAIJ_Private(Mat,IS,IS,PetscInt,MatReuse,Mat*);
#endif
