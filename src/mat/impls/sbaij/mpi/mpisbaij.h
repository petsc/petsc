
#if !defined(__MPISBAIJ_H)
#define __MPISBAIJ_H
#include "src/mat/impls/baij/seq/baij.h"
#include "src/sys/ctable.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"
#include "src/mat/impls/baij/mpi/mpibaij.h"

typedef struct {
  MPIBAIJHEADER;
  Vec           slvec0,slvec1;            /* parallel vectors */
  Vec           slvec0b,slvec1a,slvec1b;  /* seq vectors: local partition of slvec0 and slvec1 */
  VecScatter    sMvctx;                   /* scatter context for vector used for reducing communication */
} Mat_MPISBAIJ;

EXTERN PetscErrorCode MatLoad_MPISBAIJ(PetscViewer, MatType,Mat*);
#endif
