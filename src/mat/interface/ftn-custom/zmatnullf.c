#include <petsc/private/ftnimpl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define matnullspacegetvecs_     MATNULLSPACEGETVECS
  #define matnullspacerestorevecs_ MATNULLSPACERESTOREVECS
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define matnullspacegetvecs_     matnullspacegetvecs
  #define matnullspacerestorevecs_ matnullspacerestorevecs
#endif

PETSC_EXTERN void matnullspacegetvecs_(MatNullSpace *sp, PetscBool *has_const, PetscInt *n, F90Array1d *vecs, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscBool has_cnst;
  PetscInt  nv;
  Vec      *tvecs;

  CHKFORTRANNULLBOOL(has_const);
  CHKFORTRANNULLINTEGER(n);
  *ierr = MatNullSpaceGetVecs(*sp, &has_cnst, &nv, (const Vec **)&tvecs);
  if (has_const) *has_const = has_cnst;
  if (n) *n = nv;
  *ierr = F90Array1dCreate(tvecs, MPIU_FORTRANADDR, 1, nv, vecs PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void matnullspacerestorevecs_(MatNullSpace *sp, PetscBool *has_const, PetscInt *n, F90Array1d *vecs, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(vecs, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
}
