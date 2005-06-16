#include "zpetsc.h"
#include "petsc.h"

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
#if defined(PETSC_USES_CPTOFCD)
  {
  char *t = _fcdtocp(name);
  int  len1 = _fcdlen(name);
  *ierr = PetscStrncpy(t,tmp,len1);if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(name,tmp,len);if (*ierr) return;
#endif
}


EXTERN_C_END
