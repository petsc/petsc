#include <petsc/private/ftnimpl.h>
#include <petscsf.h>
#include <petscsection.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscsfdestroyremoteoffsets_ PETSCSFDESTROYREMOTEOFFSETS
  #define petscsfdistributesection_    PETSCSFDISTRIBUTESECTION
  #define petscsfcreateremoteoffsets_  PETSCSFCREATEREMOTEOFFSETS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscsfdestroyremoteoffsets_ petscsfdestroyremoteoffsets
  #define petscsfdistributesection_    petscsfdistributesection
  #define petscsfcreateremoteoffsets_  petscsfcreateremoteoffsets
#endif

PETSC_EXTERN void petscsfdestroyremoteoffsets_(F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_INT, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = PetscFree(fa);
}

PETSC_EXTERN void petscsfdistributesection_(PetscSF *sf, PetscSection *rootSection, F90Array1d *ptr, PetscSection *leafSection, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  if (ptr == PETSC_NULL_INTEGER_POINTER_Fortran) {
    *ierr = PetscSFDistributeSection(*sf, *rootSection, NULL, *leafSection);
  } else {
    PetscInt *fa;
    PetscInt  lpStart, lpEnd;

    *ierr = PetscSFDistributeSection(*sf, *rootSection, &fa, *leafSection);
    if (*ierr) return;
    *ierr = PetscSectionGetChart(*leafSection, &lpStart, &lpEnd);
    if (*ierr) return;
    *ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, lpEnd - lpStart, ptr PETSC_F90_2PTR_PARAM(ptrd));
  }
}

PETSC_EXTERN void petscsfcreateremoteoffsets_(PetscSF *pointSF, PetscSection *rootSection, PetscSection *leafSection, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *fa;
  PetscInt  lpStart, lpEnd;

  *ierr = PetscSFCreateRemoteOffsets(*pointSF, *rootSection, *leafSection, &fa);
  if (*ierr) return;
  *ierr = PetscSectionGetChart(*leafSection, &lpStart, &lpEnd);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, lpEnd - lpStart, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
