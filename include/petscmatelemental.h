#if !defined(PETSCMATELEMENTAL_H)
#define PETSCMATELEMENTAL_H

#include <petscmat.h>

#if defined(PETSC_HAVE_ELEMENTAL) && defined(__cplusplus)
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wunknown-warning-option"
#  pragma clang diagnostic ignored "-Wunused-but-set-variable"
#elif defined(__GNUC__) || defined(__GNUG__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <El.hpp>
#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#  pragma GCC diagnostic pop
#endif
#if defined(PETSC_USE_COMPLEX)
typedef El::Complex<PetscReal> PetscElemScalar;
#else
typedef PetscScalar PetscElemScalar;
#endif
#endif

#endif /* PETSCMATELEMENTAL_H */
