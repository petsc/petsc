
#if !defined(__FMDOT_H)
#include "petsc.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmdot4_      FORTRANMDOT4
#define fortranmdot3_      FORTRANMDOT3
#define fortranmdot2_      FORTRANMDOT2
#define fortranmdot1_      FORTRANMDOT1
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmdot4_      fortranmdot4
#define fortranmdot3_      fortranmdot3
#define fortranmdot2_      fortranmdot2
#define fortranmdot1_      fortranmdot1
#endif
EXTERN void fortranmdot4_(void*,void*,void*,void*,void*,PetscInt*,void*,void*,void*,void*);
EXTERN void fortranmdot3_(void*,void*,void*,void*,PetscInt*,void*,void*,void*);
EXTERN void fortranmdot2_(void*,void*,void*,PetscInt*,void*,void*);
EXTERN void fortranmdot1_(void*,void*,PetscInt*,void*);
#endif
#endif

