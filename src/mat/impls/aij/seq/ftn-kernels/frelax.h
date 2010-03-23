
#if !defined(__FRELAX_H)
#include "petscsys.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranrelaxaijforward_   FORTRANRELAXAIJFORWARD
#define fortranrelaxaijbackward_   FORTRANRELAXAIJBACKWARD
#define fortranrelaxaijforwardzero_   FORTRANRELAXAIJFORWARDZERO
#define fortranrelaxaijbackwardzero_   FORTRANRELAXAIJBACKWARDZERO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranrelaxaijforward_   fortranrelaxaijforward
#define fortranrelaxaijbackward_   fortranrelaxaijbackward
#define fortranrelaxaijforwardzero_   fortranrelaxaijforwardzero
#define fortranrelaxaijbackwardzero_   fortranrelaxaijbackwardzero
#endif
EXTERN_C_BEGIN
EXTERN void fortranrelaxaijforward_(PetscInt*,PetscReal*,void*,PetscInt*,PetscInt*,const PetscInt*,void*,void*);
EXTERN void fortranrelaxaijbackward_(PetscInt*,PetscReal*,void*,PetscInt*,PetscInt*,const PetscInt*,void*,void*);
EXTERN void fortranrelaxaijforwardzero_(PetscInt*,PetscReal*,void*,PetscInt*,PetscInt*,const PetscInt*,const void*,void*,void*);
EXTERN void fortranrelaxaijbackwardzero_(PetscInt*,PetscReal*,void*,PetscInt*,PetscInt*,const PetscInt*,const void*,void*,void*);
EXTERN_C_END
#endif
#endif

