
#if !defined(__FFT_H)
#define __FFT_H

#include <petsc-private/matimpl.h>

typedef struct {
  PetscInt       ndim;
  PetscInt       *dim;
  PetscInt       n,N;   /* local and global size of the transform */
  void           *data; /* implementation-specific data for subclass */
  PetscErrorCode (*matdestroy)(Mat);
} Mat_FFT;

EXTERN_C_BEGIN
extern PetscErrorCode MatCreate_FFTW(Mat);
EXTERN_C_END

#endif
