#if !defined(PETSCMATELEMENTAL_H)
#define PETSCMATELEMENTAL_H

#include <petscmat.h>

#if defined(PETSC_HAVE_ELEMENTAL) && defined(__cplusplus)
#include <El.hpp>
#if defined(PETSC_USE_COMPLEX)
typedef El::Complex<PetscReal> PetscElemScalar;
#else
typedef PetscScalar PetscElemScalar;
#endif
PETSC_EXTERN PetscErrorCode PetscElementalInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscElementalFinalizePackage(void);
#endif

#endif /* PETSCMATELEMENTAL_H */
