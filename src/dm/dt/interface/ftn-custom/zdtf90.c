#include <petsc/private/ftnimpl.h>
#include <petscdt.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define petscquadraturegetdata_     PETSCQUADRATUREGETDATA
  #define petscquadraturerestoredata_ PETSCQUADRATURERESTOREDATA
  #define petscquadraturesetdata_     PETSCQUADRATURESETDATA
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define petscquadraturegetdata_     petscquadraturegetdata
  #define petscquadraturerestoredata_ petscquadraturerestoredata
  #define petscquadraturesetdata_     petscquadraturesetdata
#endif

PETSC_EXTERN void petscquadraturegetdata_(PetscQuadrature *q, PetscInt *dim, PetscInt *Nc, PetscInt *npoints, F90Array1d *ptrP, F90Array1d *ptrW, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrp) PETSC_F90_2PTR_PROTO(ptrw))
{
  const PetscReal *points, *weights;
  PetscInt         dim_, Nc_, npoints_;

  CHKFORTRANNULLINTEGER(dim);
  CHKFORTRANNULLINTEGER(Nc);
  CHKFORTRANNULLINTEGER(npoints);
  *ierr = PetscQuadratureGetData(*q, &dim_, &Nc_, &npoints_, &points, &weights);
  if (*ierr) return;
  if (dim) *dim = dim_;
  if (Nc) *Nc = Nc_;
  if (npoints) *npoints = npoints_;
  if (!FORTRANNULLREALPOINTER(ptrP)) {
    *ierr = F90Array1dCreate((void *)points, MPIU_REAL, 1, npoints_ * dim_, ptrP PETSC_F90_2PTR_PARAM(ptrp));
    if (*ierr) return;
  }
  if (!FORTRANNULLREALPOINTER(ptrW)) {
    *ierr = F90Array1dCreate((void *)weights, MPIU_REAL, 1, npoints_ * Nc_, ptrW PETSC_F90_2PTR_PARAM(ptrw));
    if (*ierr) return;
  }
  *ierr = PETSC_SUCCESS;
}

PETSC_EXTERN void petscquadraturerestoredata_(PetscQuadrature *q, PetscInt *dim, PetscInt *Nc, PetscInt *npoints, F90Array1d *ptrP, F90Array1d *ptrW, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrp) PETSC_F90_2PTR_PROTO(ptrw))
{
  if (!FORTRANNULLREALPOINTER(ptrP)) {
    *ierr = F90Array1dDestroy(ptrP, MPIU_REAL PETSC_F90_2PTR_PARAM(ptrp));
    if (*ierr) return;
  }
  if (!FORTRANNULLREALPOINTER(ptrW)) {
    *ierr = F90Array1dDestroy(ptrW, MPIU_REAL PETSC_F90_2PTR_PARAM(ptrw));
    if (*ierr) return;
  }
  *ierr = PETSC_SUCCESS;
}

PETSC_EXTERN void petscquadraturesetdata_(PetscQuadrature *q, PetscInt *dim, PetscInt *Nc, PetscInt *npoints, F90Array1d *ptrP, F90Array1d *ptrW, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrp) PETSC_F90_2PTR_PROTO(ptrw))
{
  PetscReal *points, *weights;

  *ierr = F90Array1dAccess(ptrP, MPIU_REAL, (void **)&points PETSC_F90_2PTR_PARAM(ptrp));
  if (*ierr) return;
  *ierr = F90Array1dAccess(ptrW, MPIU_REAL, (void **)&weights PETSC_F90_2PTR_PARAM(ptrw));
  if (*ierr) return;
  *ierr = PetscQuadratureSetData(*q, *dim, *Nc, *npoints, points, weights);
}
