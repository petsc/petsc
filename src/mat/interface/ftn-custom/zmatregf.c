#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsettype_                      MATSETTYPE
#define matgettype_                      MATGETTYPE
#define matsetvectype_                   MATSETVECTYPE
#define matgetvectype_                   MATGETVECTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsettype_                      matsettype
#define matgettype_                      matgettype
#define matsetvectype_                   matsetvectype
#define matgetvectype_                   matgetvectype
#endif

PETSC_EXTERN void matsettype_(Mat *x,char* type_name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = MatSetType(*x,t);if (*ierr) return;
  FREECHAR(type_name,t);
}

PETSC_EXTERN void matgettype_(Mat *mm,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = MatGetType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void matsetvectype_(Mat *x,char* type_name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = MatSetVecType(*x,t);if (*ierr) return;
  FREECHAR(type_name,t);
}

PETSC_EXTERN void matgetvectype_(Mat *mm,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = MatGetVecType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}
