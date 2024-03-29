#pragma once
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/hashmapi.h>

#define MPIBAIJHEADER \
  MPIAIJHEADER; \
  PetscInt *rangebs;                            /* rmap->range/bs */ \
  PetscInt  rstartbs, rendbs, cstartbs, cendbs; /* map values / bs  */ \
  PetscInt  bs2;                                /* block size, bs2 = bs*bs */ \
  PetscInt  Mbs, Nbs;                           /* number block rows/cols in matrix; M/bs, N/bs */ \
  PetscInt  mbs, nbs;                           /* number block rows/cols on processor; m/bs, n/bs */ \
\
  /* The following variables are used for matrix assembly */ \
  PetscBool subset_off_proc_entries; /* PETSC_TRUE if assembly will always communicate a subset of the entries communicated the first time */ \
\
  /* The following variable is used by blocked matrix assembly */ \
  MatScalar *barray; /* Block array of size bs2 */ \
\
  /* Some variables to make MatSetValues and others more efficient */ \
  PetscInt    rstart_bs, rend_bs; \
  PetscInt    cstart_bs, cend_bs; \
  PetscInt   *ht; /* Hash table to speed up matrix assembly */ \
  MatScalar **hd; /* Hash table data */ \
  PetscInt    ht_size; \
  PetscInt    ht_total_ct, ht_insert_ct; /* Hash table statistics */ \
  PetscBool   ht_flag;                   /* Flag to indicate if hash tables are used */ \
  double      ht_fact;                   /* Factor to determine the HT size */ \
\
  PetscInt   setvalueslen;  /* only used for single precision computations */ \
  MatScalar *setvaluescopy; /* area double precision values in MatSetValuesXXX() are copied*/ \
                            /* before calling MatSetValuesXXX_MPIBAIJ_MatScalar() */ \
  PetscBool      ijonly;    /* used in  MatCreateSubMatrices_MPIBAIJ_local() for getting ij structure only */ \
  struct _MatOps cops

typedef struct {
  MPIBAIJHEADER;
} Mat_MPIBAIJ;

PETSC_INTERN PetscErrorCode MatView_MPIBAIJ(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatLoad_MPIBAIJ(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatView_MPIBAIJ_Binary(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatLoad_MPIBAIJ_Binary(Mat, PetscViewer);

PETSC_INTERN PetscErrorCode MatCreateColmap_MPIBAIJ_Private(Mat);
PETSC_INTERN PetscErrorCode MatCreateSubMatrices_MPIBAIJ(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *[]);
PETSC_INTERN PetscErrorCode MatCreateSubMatrices_MPIBAIJ_local(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *, PetscBool);
PETSC_INTERN PetscErrorCode MatCreateSubMatrix_MPIBAIJ_Private(Mat, IS, IS, PetscInt, MatReuse, Mat *, PetscBool);
PETSC_INTERN PetscErrorCode MatGetMultiProcBlock_MPIBAIJ(Mat, MPI_Comm, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatIncreaseOverlap_MPIBAIJ(Mat, PetscInt, IS[], PetscInt);
PETSC_INTERN PetscErrorCode MatIncreaseOverlap_MPIBAIJ_Once(Mat, PetscInt, IS *);
PETSC_INTERN PetscErrorCode MatMPIBAIJSetPreallocation_MPIBAIJ(Mat B, PetscInt bs, PetscInt d_nz, const PetscInt *d_nnz, PetscInt o_nz, const PetscInt *o_nnz);
PETSC_INTERN PetscErrorCode MatAXPYGetPreallocation_MPIBAIJ(Mat, const PetscInt *, Mat, const PetscInt *, PetscInt *);

PETSC_INTERN PetscErrorCode MatConjugate_SeqBAIJ(Mat);
