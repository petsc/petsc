#include "private/fortranimpl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectgetname_        PETSCOBJECTGETNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectgetname_        petscobjectgetname
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscobjectgetname_(PetscObject *obj,CHAR name PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tmp;
  *ierr = PetscObjectGetName(*obj,&tmp);
  *ierr = PetscStrncpy(name,tmp,len);if (*ierr) return;
}


EXTERN_C_END
