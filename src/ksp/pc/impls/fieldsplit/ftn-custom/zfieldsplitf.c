#include <petsc/private/ftnimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcfieldsplitgetsubksp_          PCFIELDSPLITGETSUBKSP
  #define pcfieldsplitschurgetsubksp_     PCFIELDSPLITSCHURGETSUBKSP
  #define pcfieldsplitrestoresubksp_      PCFIELDSPLITRESTORESUBKSP
  #define pcfieldsplitschurrestoresubksp_ PCFIELDSPLITSCHURRESTORESUBKSP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcfieldsplitgetsubksp_          pcfieldsplitgetsubksp
  #define pcfieldsplitschurgetsubksp_     pcfieldsplitschurgetsubksp
  #define pcfieldsplitrestoresubksp_      pcfieldsplitrestoresubksp
  #define pcfieldsplitschurrestoresubksp_ pcfieldsplitschurrestoresubksp
#endif

PETSC_EXTERN void pcfieldsplitschurgetsubksp_(PC *pc, PetscInt *n_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  KSP     *tksp;
  PetscInt nloc;
  CHKFORTRANNULLINTEGER(n_local);
  *ierr = PCFieldSplitSchurGetSubKSP(*pc, &nloc, &tksp);
  if (*ierr) return;
  if (n_local) *n_local = nloc;
  *ierr = F90Array1dCreate(tksp, MPIU_FORTRANADDR, 1, nloc, ksp PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void pcfieldsplitgetsubksp_(PC *pc, PetscInt *n_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  KSP     *tksp;
  PetscInt nloc;
  CHKFORTRANNULLINTEGER(n_local);
  *ierr = PCFieldSplitGetSubKSP(*pc, &nloc, &tksp);
  if (*ierr) return;
  if (n_local) *n_local = nloc;
  *ierr = F90Array1dCreate(tksp, MPIU_FORTRANADDR, 1, nloc, ksp PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void pcfieldsplitrestoresubksp_(PC *pc, PetscInt *n_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  void *array;
  *ierr = F90Array1dAccess(ksp, MPIU_FORTRANADDR, (void **)&array PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ksp, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = PetscFree(array);
}

PETSC_EXTERN void pcfieldsplitschurerestoresubksp_(PC *pc, PetscInt *n_local, F90Array1d *ksp, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  void *array;
  *ierr = F90Array1dAccess(ksp, MPIU_FORTRANADDR, (void **)&array PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ksp, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = PetscFree(array);
}
