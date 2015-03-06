#if !defined(__PETSCMATELEMENTAL_H)
#define __PETSCMATELEMENTAL_H

#include <petscmat.h>

#if defined(PETSC_HAVE_ELEMENTAL) && defined(__cplusplus)
#include <El.hpp>
/* c++ prototypes requiring elemental datatypes. */
PETSC_EXTERN PetscErrorCode MatElementalHermitianGenDefEig(El::Pencil,El::UpperOrLower,Mat,Mat,Mat*,Mat*,El::SortType,El::HermitianEigSubset<PetscElemScalar>,const El::HermitianEigCtrl<PetscElemScalar>);
#endif

#endif /* __PETSCMATELEMENTAL_H */
