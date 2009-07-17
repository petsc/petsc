
#if !defined(__FMULATCRL_H)
#include "petsc.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultcrl_      FORTRANMULTCRL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultcrl_      fortranmultcrl
#endif
EXTERN void fortranmultcrl_(PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*);
#endif
#endif

