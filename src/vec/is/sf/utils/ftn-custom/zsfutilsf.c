#include <petsc/private/fortranimpl.h>
#include <petscsf.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscsfdistributesection_ PETSCSFDISTRIBUTESECTION
  #define petscsfgetgraphlayout_    PETSCSFGETGRAPHLAYOUT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscsfdistributesection_ petscsfdistributesection
  #define petscsfgetgraphlayout_    petscsfgetgraphlayout
#endif

PETSC_EXTERN void petscsfdistributesection_(PetscSF *sf, PetscSection *rootSection, PetscInt **remoteOffsets, PetscSection *leafSection, PetscErrorCode *__ierr)
{
  if (remoteOffsets != PETSC_NULL_INTEGER_Fortran) {
    (void)PetscError(PETSC_COMM_SELF, __LINE__, "PetscSFDistributeSection_Fortran", __FILE__, PETSC_ERR_SUP, PETSC_ERROR_INITIAL, "The remoteOffsets argument must be PETSC_NULL_INTEGER in Fortran");
    *__ierr = PETSC_ERR_SUP;
    return;
  }
  *__ierr = PetscSFDistributeSection(*sf, *rootSection, NULL, *leafSection);
}

PETSC_EXTERN void petscsfgetgraphlayout_(PetscSF *sf, PetscLayout *layout, PetscInt *nleaves, const PetscInt *ilocal[], PetscInt *gremote[], PetscErrorCode *__ierr)
{
  *__ierr = PetscSFGetGraphLayout(*sf, layout, nleaves, ilocal, gremote);
}
