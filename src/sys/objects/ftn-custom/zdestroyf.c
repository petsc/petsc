#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjecttypecompare_          PETSCOBJECTTYPECOMPARE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjecttypecompare_          petscobjecttypecompare
#endif

PETSC_EXTERN void petscobjecttypecompare_(PetscObject *obj,char* type_name,
                                     PetscBool  *same,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;
  FIXCHAR(type_name,len,c1);
  *ierr = PetscObjectTypeCompare(*obj,c1,same);if (*ierr) return;
  FREECHAR(type_name,c1);
}

