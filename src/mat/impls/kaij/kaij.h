#ifndef _KAIJ_H
#define _KAIJ_H

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#define KAIJHEADER \
  PetscInt     p, q; \
  Mat          AIJ; \
  PetscScalar *S; \
  PetscScalar *T; \
  PetscScalar *ibdiag; \
  PetscBool    ibdiagvalid, getrowactive, isTI; \
  struct { \
    PetscBool    setup; \
    PetscScalar *w, *work, *t, *arr, *y; \
  } sor;

typedef struct {
  KAIJHEADER
} Mat_SeqKAIJ;

typedef struct {
  KAIJHEADER
  Mat              OAIJ;  /* sequential KAIJ matrix that corresponds to off-diagonal matrix entries (diagonal entries are stored in 'AIJ') */
  Mat              A;     /* AIJ matrix describing the blockwise action of the KAIJ matrix; compare with struct member 'AIJ' in sequential case */
  VecScatter       ctx;   /* update ghost points for parallel case */
  Vec              w;     /* work space for ghost values for parallel case */
  PetscObjectState state; /* state of the matrix A when AIJ and OIJ were last updated */
} Mat_MPIKAIJ;

#endif
