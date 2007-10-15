#include "private/zpetsc.h"
#include "petscpc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcfactorsetmatorderingtype_       PCFACTORSETMATORDERINGTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcfactorsetmatorderingtype_       pcfactorsetmatorderingtype
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pcfactorsetmatorderingtype_(PC *pc,CHAR ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;

    FIXCHAR(ordering,len,t);
    *ierr = PCFactorSetMatOrderingType(*pc,t);
    FREECHAR(ordering,t);
}

EXTERN_C_END
