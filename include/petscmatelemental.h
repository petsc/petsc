#pragma once

#include <petscmat.h>

/* SUBMANSEC = Mat */

#if defined(PETSC_HAVE_ELEMENTAL) && defined(__cplusplus)
  #if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunknown-warning-option"
    #pragma clang diagnostic ignored "-Wunused-parameter"
    #pragma clang diagnostic ignored "-Wunused-but-set-variable"
    #pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
    #pragma clang diagnostic ignored "-Wextra-semi"
  #elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    #pragma GCC diagnostic ignored "-Wextra-semi"
  #endif
  #include <El.hpp>
  #if defined(__clang__)
    #pragma clang diagnostic pop
  #elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic pop
  #endif
  #if defined(PETSC_USE_COMPLEX)
typedef El::Complex<PetscReal> PetscElemScalar;
  #else
/*MC
  PetscElemScalar - Scalar datatype used when interfacing PETSc to the Elemental dense linear algebra library through `MATELEMENTAL`

  Level: developer

  Note:
  When PETSc is configured with complex scalars, `PetscElemScalar` is `El::Complex<PetscReal>` so that PETSc complex values can be passed to Elemental without conversion.
  In the real case it is the same as `PetscScalar`.

.seealso: `PetscScalar`, `MATELEMENTAL`, `MatSetType()`
M*/
typedef PetscScalar PetscElemScalar;
  #endif
#endif
