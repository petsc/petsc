#include <petsc/private/ftnimpl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matnullspacegetvecs_ MATNULLSPACEGETVECS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matnullspacegetvecs_ matnullspacegetvecs
#endif

PETSC_EXTERN void matnullspacegetvecs_(MatNullSpace *sp, PetscBool *HAS_CNST, PetscInt *N, Vec *VECS, PetscErrorCode *ierr)
{
  PetscBool  has_cnst;
  PetscInt   i, n;
  const Vec *vecs;

  CHKFORTRANNULLBOOL(HAS_CNST);
  CHKFORTRANNULLINTEGER(N);
  CHKFORTRANNULLOBJECT(VECS);

  *ierr = MatNullSpaceGetVecs(*sp, &has_cnst, &n, &vecs);

  if (HAS_CNST) { *HAS_CNST = has_cnst; }
  if (N) { *N = n; }
  if (VECS) {
    for (i = 0; i < n; i++) { VECS[i] = vecs[i]; }
  }
}
