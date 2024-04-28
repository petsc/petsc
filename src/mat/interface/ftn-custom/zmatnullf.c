#include <petsc/private/fortranimpl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matnullspacecreate0_ MATNULLSPACECREATE0
  #define matnullspacecreate1_ MATNULLSPACECREATE1
  #define matnullspacegetvecs_ MATNULLSPACEGETVECS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matnullspacecreate0_ matnullspacecreate0
  #define matnullspacecreate1_ matnullspacecreate1
  #define matnullspacegetvecs_ matnullspacegetvecs
#endif

PETSC_EXTERN void matnullspacecreate0_(MPI_Fint *comm, PetscBool *has_cnst, PetscInt *n, Vec vecs[], MatNullSpace *SP, PetscErrorCode *ierr)
{
  *ierr = MatNullSpaceCreate(MPI_Comm_f2c(*(comm)), *has_cnst, *n, vecs, SP);
}

PETSC_EXTERN void matnullspacecreate1_(MPI_Fint *comm, PetscBool *has_cnst, PetscInt *n, Vec vecs[], MatNullSpace *SP, PetscErrorCode *ierr)
{
  *ierr = MatNullSpaceCreate(MPI_Comm_f2c(*(comm)), *has_cnst, *n, vecs, SP);
}

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
