#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectsetname_        PETSCOBJECTSETNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectsetname_        petscobjectsetname
#endif

PETSC_EXTERN void petscobjectsetname_(PetscObject *obj,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t1;

  FIXCHAR(name,len,t1);
  *ierr = PetscObjectSetName(*obj,t1);if (*ierr) return;
  FREECHAR(name,t1);
}

