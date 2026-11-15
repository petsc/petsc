#pragma once

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#define KAIJHEADER \
  PetscInt         p, q; \
  Mat              AIJ; \
  PetscScalar     *S; \
  PetscScalar     *T; \
  PetscScalar     *ibdiag; \
  PetscInt         ibdiagcount;  /* allocated length of ibdiag */ \
  PetscObjectState ibdiagstate;  /* state of A when ibdiag was last computed */ \
  PetscObjectState ibdiagkstate; /* state of the KAIJ matrix when ibdiag was last computed */ \
  PetscBool        ibdiagvalid, getrowactive, isTI; \
  struct { \
    PetscBool    setup; \
    PetscScalar *w, *work, *t, *arr, *y; \
  } sor;

typedef struct {
  KAIJHEADER
} Mat_SeqKAIJ;

typedef struct {
  KAIJHEADER
  Mat              OAIJ;         /* sequential KAIJ matrix that corresponds to off-diagonal matrix entries (diagonal entries are stored in 'AIJ') */
  Mat              A;            /* AIJ matrix describing the blockwise action of the KAIJ matrix; compare with struct member 'AIJ' in sequential case */
  VecScatter       ctx;          /* update ghost points for parallel case */
  Vec              w;            /* work space for ghost values for parallel case */
  PetscInt        *ghosts;       /* copy of the off-diagonal ghost columns of A that 'ctx' and 'w' were built from */
  PetscInt         nghosts;      /* length of 'ghosts' */
  PetscObjectState state;        /* state of the matrix A when AIJ and OAIJ were last updated */
  PetscObjectState kstate;       /* state of the KAIJ matrix when the submatrices were last updated */
  PetscBool        scatterstale; /* 'ctx' must be rebuilt before the next MatMult() */
} Mat_MPIKAIJ;
