#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjecttypecompare_          PETSCOBJECTTYPECOMPARE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjecttypecompare_          petscobjecttypecompare
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscobjecttypecompare_(PetscObject *obj,CHAR type_name PETSC_MIXED_LEN(len),
                                     PetscBool  *same,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;
  FIXCHAR(type_name,len,c1);
  *ierr = PetscObjectTypeCompare(*obj,c1,same);
  FREECHAR(type_name,c1);
}


EXTERN_C_END
