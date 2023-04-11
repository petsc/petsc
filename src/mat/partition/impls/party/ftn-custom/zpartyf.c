#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matpartitioningpartysetglobal_ MATPARTITIONINGPARTYSETGLOBAL
  #define matpartitioningpartysetlocal_  MATPARTITIONINGPARTYSETLOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matpartitioningpartysetglobal_ matpartitioningpartysetglobal
  #define matpartitioningpartysetlocal_  matpartitioningpartysetlocal
#endif

PETSC_EXTERN void matpartitioningpartysetglobal_(MatPartitioning *part, char *method, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(method, len, t);
  *ierr = MatPartitioningPartySetGlobal(*part, t);
  if (*ierr) return;
  FREECHAR(method, t);
}

PETSC_EXTERN void matpartitioningpartysetlocal_(MatPartitioning *part, char *method, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(method, len, t);
  *ierr = MatPartitioningPartySetLocal(*part, t);
  if (*ierr) return;
  FREECHAR(method, t);
}
