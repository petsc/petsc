#include <petsc/private/fortranimpl.h>
#include <petscpartitioner.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscpartitionersettype_         PETSCPARTITIONERSETTYPE
  #define petscpartitionergettype_         PETSCPARTITIONERGETTYPE
  #define petscpartitionerviewfromoptions_ PETSCPARTITIONERVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define petscpartitionersettype_         petscpartitionersettype
  #define petscpartitionergettype_         petscpartitionergettype
  #define petscpartitionerviewfromoptions_ petscpartitionerviewfromoptions
#endif

PETSC_EXTERN void petscpartitionergettype_(PetscPartitioner *mm, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PetscPartitionerGetType(*mm, &tname);
  if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name, tname, len);
    if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}

PETSC_EXTERN void petscpartitionerviewfromoptions_(PetscPartitioner *part, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscPartitionerViewFromOptions(*part, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}

PETSC_EXTERN void petscpartitionersettype_(PetscPartitioner *x, char *type_name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type_name, len, t);
  *ierr = PetscPartitionerSetType(*x, t);
  FREECHAR(type_name, t);
}
