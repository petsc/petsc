#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectgettype_        PETSCOBJECTGETTYPE
#define petscobjectsettype_        PETSCOBJECTSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectgettype_        petscobjectgettype
#define petscobjectsettype_        petscobjectsettype
#endif

PETSC_EXTERN void petscobjectgettype_(PetscObject *obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tmp;
  *ierr = PetscObjectGetType(*obj,&tmp);if (*ierr) return;
  *ierr = PetscStrncpy(type,tmp,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,type,len);
}

PETSC_EXTERN void petscobjectsettype_(PetscObject *obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t1;

  FIXCHAR(type,len,t1);
  *ierr = PetscObjectSetType(*obj,t1);if (*ierr) return;
  FREECHAR(type,t1);
}

