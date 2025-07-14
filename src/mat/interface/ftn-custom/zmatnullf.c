#include <petsc/private/ftnimpl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matnullspacegetvecs_     MATNULLSPACEGETVECS
  #define matnullspacerestorevecs_ MATNULLSPACERESTOREVECS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matnullspacegetvecs_     matnullspacegetvecs
  #define matnullspacerestorevecs_ matnullspacerestorevecs
#endif

PETSC_EXTERN void matnullspacegetvecs_(MatNullSpace *sp, PetscBool *HAS_CNST, PetscInt *N, F90Array1d *vecs, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscBool has_cnst;
  PetscInt  n;
  Vec      *tvecs;

  CHKFORTRANNULLBOOL(HAS_CNST);
  CHKFORTRANNULLINTEGER(N);
  *ierr = MatNullSpaceGetVecs(*sp, &has_cnst, &n, (const Vec **)&tvecs);
  if (HAS_CNST) *HAS_CNST = has_cnst;
  if (N) *N = n;
  *ierr = F90Array1dCreate(tvecs, MPIU_FORTRANADDR, 1, n, vecs PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void matnullspacerestorevecs_(MatNullSpace *sp, PetscBool *HAS_CNST, PetscInt *N, F90Array1d *vecs, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(vecs, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
}
