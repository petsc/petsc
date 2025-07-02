#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcbjacobigetsubksp_     PCBJACOBIGETSUBKSP
  #define pcbjacobirestoresubksp_ PCBJACOBIRESTORESUBKSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcbjacobigetsubksp_     pcbjacobigetsubksp
  #define pcbjacobirestoresubksp_ pcbjacobirestoresubksp
#endif

PETSC_EXTERN void pcbjacobigetsubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  KSP     *tksp;
  PetscInt nloc, flocal;

  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCBJacobiGetSubKSP(*pc, &nloc, &flocal, &tksp);
  if (n_local) *n_local = nloc;
  if (first_local) *first_local = flocal;
  *ierr = F90Array1dCreate(tksp, MPIU_FORTRANADDR, 1, nloc, ksp PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void pcbjacobirestoresubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ksp, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
}
