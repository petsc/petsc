#include <petsc/private/fortranimpl.h>
#include <petscds.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdsgettabulation_        PETSCDSGETTABULATION
#define petscdsrestoretabulation_    PETSCDSRESTORETABULATION
#define petscdsgetbdtabulation_      PETSCDSGETBDTABULATION
#define petscdsrestorebdtabulation_  PETSCDSRESTOREBDTABULATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdsgettabulation_        petscdsgettabulation
#define petscdsrestoretabulation_    petscdsrestoretabulation
#define petscdsgetbdtabulation_      petscdsgetbdtabulation
#define petscdsrestorebdtabulation_  petscdsrestorebdtabulation
#endif

PETSC_EXTERN void petscdsgettabulation_(PetscDS *prob, PetscInt *f, F90Array1d *ptrB, F90Array1d *ptrD, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrb) PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFE          fe;
  PetscQuadrature  q;
  PetscInt         dim, Nb, Nc, Nq;
  PetscTabulation *T;

  *ierr = PetscDSGetSpatialDimension(*prob, &dim);if (*ierr) return;
  *ierr = PetscDSGetDiscretization(*prob, *f, (PetscObject *) &fe);if (*ierr) return;
  *ierr = PetscFEGetDimension(fe, &Nb);if (*ierr) return;
  *ierr = PetscFEGetNumComponents(fe, &Nc);if (*ierr) return;
  *ierr = PetscFEGetQuadrature(fe, &q);if (*ierr) return;
  *ierr = PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL);if (*ierr) return;
  *ierr = PetscDSGetTabulation(*prob, &T);if (*ierr) return;
  *ierr = F90Array1dCreate((void *) T[*f]->T[0], MPIU_REAL, 1, Nq*Nb*Nc,     ptrB PETSC_F90_2PTR_PARAM(ptrb));if (*ierr) return;
  *ierr = F90Array1dCreate((void *) T[*f]->T[1], MPIU_REAL, 1, Nq*Nb*Nc*dim, ptrD PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petscdsrestoretabulation_(PetscDS *prob, PetscInt *f, F90Array1d *ptrB, F90Array1d *ptrD, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrb) PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptrB, MPIU_REAL PETSC_F90_2PTR_PARAM(ptrb));if (*ierr) return;
  *ierr = F90Array1dDestroy(ptrD, MPIU_REAL PETSC_F90_2PTR_PARAM(ptrd));
}
