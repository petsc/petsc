
#ifndef __FFT_H
#define __FFT_H

#include <petsc/private/matimpl.h>

typedef struct {
  PetscInt  ndim;
  PetscInt *dim;
  PetscInt  n, N; /* local and global size of the transform */
  void     *data; /* implementation-specific data for subclass */

  PetscErrorCode (*matdestroy)(Mat);
} Mat_FFT;

PETSC_EXTERN PetscErrorCode MatCreate_FFTW(Mat);

#endif
