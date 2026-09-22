#include <petsc/private/ftnimpl.h>
#include <petscvec.h>
#include <petscviewer.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define vecgetownershipranges_     VECGETOWNERSHIPRANGES
  #define vecrestoreownershipranges_ VECRESTOREOWNERSHIPRANGES
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define vecgetownershipranges_     vecgetownershipranges
  #define vecrestoreownershipranges_ vecrestoreownershipranges
#endif

PETSC_EXTERN void vecgetownershipranges_(Vec *x, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *fa;
  PetscMPIInt size;

  *ierr = VecGetOwnershipRanges(*x, (const PetscInt **)&fa);
  if (*ierr) return;
  MPI_Comm_size(PetscObjectComm((PetscObject)*x), &size);
  *ierr = F90Array1dCreate(fa, MPIU_INT, 1, size + 1, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecrestoreownershipranges_(Vec *x, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
}
