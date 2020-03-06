#include <petsc/private/fortranimpl.h>
#include <petscdt.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscquadraturegetdata_      PETSCQUADRATUREGETDATA
#define petscquadraturerestoredata_  PETSCQUADRATURERESTOREDATA
#define petscquadraturesetdata_      PETSCQUADRATURESETDATA
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscquadraturegetdata_      petscquadraturegetdata
#define petscquadraturerestoredata_  petscquadraturerestoredata
#define petscquadraturesetdata_      petscquadraturesetdata
#endif

PETSC_EXTERN void petscquadraturegetdata_(PetscQuadrature *q, PetscInt *dim, PetscInt *Nc, PetscInt *npoints, F90Array1d *ptrP, F90Array1d *ptrW, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrp) PETSC_F90_2PTR_PROTO(ptrw))
{
  const PetscReal *points, *weights;

  *ierr = PetscQuadratureGetData(*q, dim, Nc, npoints, &points, &weights);if (*ierr) return;
  *ierr = F90Array1dCreate((void *) points, MPIU_REAL, 1, (*npoints)*(*dim), ptrP PETSC_F90_2PTR_PARAM(ptrp));if (*ierr) return;
  *ierr = F90Array1dCreate((void *) weights, MPIU_REAL, 1, (*npoints)*(*Nc), ptrW PETSC_F90_2PTR_PARAM(ptrw));
}

PETSC_EXTERN void petscquadraturerestoredata_(PetscQuadrature *q, PetscInt *dim, PetscInt *Nc, PetscInt *npoints, F90Array1d *ptrP, F90Array1d *ptrW, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrp) PETSC_F90_2PTR_PROTO(ptrw))
{
  *ierr = F90Array1dDestroy(ptrP, MPIU_REAL PETSC_F90_2PTR_PARAM(ptrp));if (*ierr) return;
  *ierr = F90Array1dDestroy(ptrW, MPIU_REAL PETSC_F90_2PTR_PARAM(ptrw));
}

PETSC_EXTERN void petscquadraturesetdata_(PetscQuadrature *q, PetscInt *dim, PetscInt *Nc, PetscInt *npoints, F90Array1d *ptrP, F90Array1d *ptrW, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrp) PETSC_F90_2PTR_PROTO(ptrw))
{
  PetscReal *points, *weights;

  *ierr = F90Array1dAccess(ptrP, MPIU_REAL, (void **) &points PETSC_F90_2PTR_PARAM(ptrp));if (*ierr) return;
  *ierr = F90Array1dAccess(ptrW, MPIU_REAL, (void **) &weights PETSC_F90_2PTR_PARAM(ptrw));if (*ierr) return;
  *ierr = PetscQuadratureSetData(*q, *dim, *Nc, *npoints, points, weights);
}
