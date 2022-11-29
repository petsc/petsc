#ifndef PETSCDEVICE_CUPM_H
#define PETSCDEVICE_CUPM_H

#include <petscmacros.h>
#include <petscdevice_cuda.h>
#include <petscdevice_hip.h>

#if PetscDefined(USING_HCC) && PetscDefined(USING_NVCC)
  #error using both nvcc and hipcc at the same time?
#endif

#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  #define PETSC_HAVE_CUPM 1
#endif

#if PetscDefined(USING_NVCC) || PetscDefined(USING_HCC)
  #define PETSC_USING_CUPMCC 1
#endif

#endif // PETSCDEVICE_CUPM_H
