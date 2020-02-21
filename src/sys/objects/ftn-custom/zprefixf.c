#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectsetoptionsprefix     PETSCOBJECTSETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectsetoptionsprefix_    petscobjectsetoptionsprefix
#endif

PETSC_EXTERN void petscobjectsetoptionsprefix_(PetscObject *obj,char* prefix,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PetscObjectSetOptionsPrefix(*obj,t);if (*ierr) return;
  FREECHAR(prefix,t);
}



