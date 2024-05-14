#include <petsc/private/fortranimpl.h>
#include <petscpartitioner.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscpartitionerviewfromoptions_ PETSCPARTITIONERVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define petscpartitionerviewfromoptions_ petscpartitionerviewfromoptions
#endif

PETSC_EXTERN void petscpartitionerviewfromoptions_(PetscPartitioner *part, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscPartitionerViewFromOptions(*part, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
