#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcbjacobigetsubksp_ PCBJACOBIGETSUBKSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcbjacobigetsubksp_ pcbjacobigetsubksp
#endif

PETSC_EXTERN void pcbjacobigetsubksp_(PC *pc, PetscInt *n_local, PetscInt *first_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  KSP     *tksp;
  PetscInt nloc, flocal;
  size_t  *iksp;

  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCBJacobiGetSubKSP(*pc, &nloc, &flocal, &tksp);
  if (n_local) *n_local = nloc;
  if (first_local) *first_local = flocal;
  *ierr = F90Array1dAccess(ksp, MPIU_FORTRANADDR, (void **)&iksp PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  if (!iksp || *iksp == 0) { *ierr = F90Array1dCreate(tksp, MPIU_FORTRANADDR, 1, nloc, ksp PETSC_F90_2PTR_PARAM(ptrd)); }
}
