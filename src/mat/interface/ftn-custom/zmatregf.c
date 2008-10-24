#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsettype_                      MATSETTYPE
#define matgettype_                      MATGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsettype_                      matsettype
#define matgettype_                      matgettype
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL matsettype_(Mat *x,CHAR type_name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = MatSetType(*x,t);
  FREECHAR(type_name,t);
}

void PETSC_STDCALL matgettype_(Mat *mm,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = MatGetType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}


EXTERN_C_END
