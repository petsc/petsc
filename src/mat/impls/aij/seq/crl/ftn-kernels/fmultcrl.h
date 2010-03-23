
#if !defined(__FMULATCRL_H)
#include "petscsys.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultcrl_      FORTRANMULTCRL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultcrl_      fortranmultcrl
#endif
EXTERN_C_BEGIN
EXTERN void fortranmultcrl_(PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*);
EXTERN_C_END
#endif
#endif

