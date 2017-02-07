#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectgettype_        PETSCOBJECTGETTYPE
#define petscobjectsettype_        PETSCOBJECTSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectgettype_        petscobjectgettype
#define petscobjectsettype_        petscobjectsettype
#endif

PETSC_EXTERN void PETSC_STDCALL petscobjectgettype_(PetscObject *obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tmp;
  *ierr = PetscObjectGetType(*obj,&tmp);if (*ierr) return;
  *ierr = PetscStrncpy(type,tmp,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,type,len);
}

PETSC_EXTERN void PETSC_STDCALL petscobjectsettype_(PetscObject *obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;

  FIXCHAR(type,len,t1);
  *ierr = PetscObjectSetType(*obj,t1);
  FREECHAR(type,t1);
}

