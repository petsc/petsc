
#if !defined(__SGEMV_H)
#include <petscsys.h>
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define msgemv_  MSGEMV
#define msgemvp_ MSGEMVP
#define msgemvm_ MSGEMVM
#define msgemvt_ MSGEMVT
#define msgemmi_ MSGEMMI
#define msgemm_  MSGEMM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define msgemv_  msgemv
#define msgemvp_ msgemvp
#define msgemvm_ msgemvm
#define msgemvt_ msgemvt
#define msgemmi_ msgemmi
#define msgemm_  msgemm
#endif
EXTERN_C_BEGIN
extern void msgemv_(PetscInt*,PetscInt *,MatScalar*,PetscScalar*,PetscScalar*);
extern void msgemvp_(PetscInt*,PetscInt *,MatScalar*,PetscScalar*,PetscScalar*);
extern void msgemvm_(PetscInt*,PetscInt *,MatScalar*,PetscScalar*,PetscScalar*);
extern void msgemvt_(PetscInt*,PetscInt *,MatScalar*,PetscScalar*,PetscScalar*);
extern void msgemmi_(PetscInt*,MatScalar*,MatScalar*,MatScalar*);
extern void msgemm_(PetscInt*,MatScalar*,MatScalar*,MatScalar*);
EXTERN_C_END
#endif

