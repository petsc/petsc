#ifndef _matcliqueimpl_h
#define _matcliqueimpl_h

#include <clique.hpp>
#include <petsc-private/matimpl.h>

#if defined (PETSC_USE_COMPLEX)
typedef cliq::Complex<PetscReal> PetscCliqScalar;
#else
typedef PetscScalar PetscCliqScalar;
#endif

typedef struct {
} Mat_Clique;

#endif
