#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcbddccreatefetidpoperators_ PCBDDCCREATEFETIDPOPERATORS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcbddccreatefetidpoperators_ pcbddccreatefetidpoperators
#endif

PETSC_EXTERN void pcbddccreatefetidpoperators_(PC *pc, PetscBool *fully_redundant, char *prefix, Mat *fetidp_mat, PC *fetidp_pc, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(prefix, len, t);
  *ierr = PCBDDCCreateFETIDPOperators(*pc, *fully_redundant, t, fetidp_mat, fetidp_pc);
  if (*ierr) return;
  FREECHAR(prefix, t);
}
