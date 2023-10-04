#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcsettype_ PCSETTYPE
  #define pcgettype_ PCGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcsettype_ pcsettype
  #define pcgettype_ pcgettype
#endif

PETSC_EXTERN void pcsettype_(PC *pc, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  *ierr = PCSetType(*pc, t);
  FREECHAR(type, t);
}

PETSC_EXTERN void pcgettype_(PC *pc, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PCGetType(*pc, &tname);
  *ierr = PetscStrncpy(name, tname, len);
  if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}
