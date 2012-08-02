
#include <clique.hpp>
#include <petsc-private/matimpl.h>

#if defined (PETSC_USE_COMPLEX)
typedef cliq::Complex<PetscReal> PetscCliqScalar;
#else
typedef PetscScalar PetscCliqScalar;
#endif

typedef struct {
  cliq::DistSparseMatrix<PetscCliqScalar> *cmat;
} Mat_Clique;
