#pragma once
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

typedef struct {
  MPIBAIJHEADER;
  Vec        slvec0, slvec1;            /* parallel vectors */
  Vec        slvec0b, slvec1a, slvec1b; /* seq vectors: local partition of slvec0 and slvec1 */
  VecScatter sMvctx;                    /* scatter context for vector used for reducing communication */

  Vec diag; /* used in MatSOR_MPISBAIJ() with Eisenstat */
  Vec bb1, xx1;

  /* these are used in MatSetValues() as tmp space before passing to the stasher */
  PetscInt   n_loc, *in_loc; /* nloc is length of in_loc and v_loc */
  MatScalar *v_loc;
} Mat_MPISBAIJ;

PETSC_INTERN PetscErrorCode MatSetUpMultiply_MPISBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatDisAssemble_MPISBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatIncreaseOverlap_MPISBAIJ(Mat, PetscInt, IS[], PetscInt);
