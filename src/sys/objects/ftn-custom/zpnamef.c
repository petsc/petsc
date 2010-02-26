#include "private/fortranimpl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectsetname_        PETSCOBJECTSETNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectsetname_        petscobjectsetname
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscobjectsetname_(PetscObject *obj,CHAR name PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;

  FIXCHAR(name,len,t1);
  *ierr = PetscObjectSetName(*obj,t1);
  FREECHAR(name,t1);
}

EXTERN_C_END
