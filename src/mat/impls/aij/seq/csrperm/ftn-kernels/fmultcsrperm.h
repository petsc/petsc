
#if !defined(__FNORM_H)
#include "petsc.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCRL)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultcrl_      FORTRANMULTCRL
#define fortranmultcsrperm_  FORTRANMULTCSRPERM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultcrl_      fortranmultcrl
#define fortranmultcsrperm_  fortranmultcsrperm
#endif
EXTERN void fortranmultcrl_(PetscInt*,PetscInt*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*);
EXTERN void fortranmultcsrperm_(PetscInt*,PetscScalar*,PetscInt*,PetscInt*,PetscScalar*,PetscScalar*);
#endif
#endif

