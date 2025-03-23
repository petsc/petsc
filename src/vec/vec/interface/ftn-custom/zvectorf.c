#include <petsc/private/ftnimpl.h>
#include <petscvec.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define vecsetvalue_               VECSETVALUE
  #define vecsetvaluelocal_          VECSETVALUELOCAL
  #define vecgetownershipranges_     VECGETOWNERSHIPRANGES
  #define vecrestoreownershipranges_ VECRESTOREOWNERSHIPRANGES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define vecsetvalue_               vecsetvalue
  #define vecsetvaluelocal_          vecsetvaluelocal
  #define vecgetownershipranges_     vecgetownershipranges
  #define vecrestoreownershipranges_ vecrestoreownershipranges
#endif

PETSC_EXTERN void vecsetvalue_(Vec *v, PetscInt *i, PetscScalar *va, InsertMode *mode, PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses PetscCall() which has a return in it */
  *ierr = VecSetValues(*v, 1, i, va, *mode);
}

PETSC_EXTERN void vecsetvaluelocal_(Vec *v, PetscInt *i, PetscScalar *va, InsertMode *mode, PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses PetscCall() which has a return in it */
  *ierr = VecSetValuesLocal(*v, 1, i, va, *mode);
}

PETSC_EXTERN void vecgetownershipranges_(Vec *v, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt   *fa;
  PetscMPIInt size;

  *ierr = VecGetOwnershipRanges(*v, (const PetscInt **)&fa);
  if (*ierr) return;
  MPI_Comm_size(PetscObjectComm((PetscObject)*v), &size);
  *ierr = F90Array1dCreate(fa, MPIU_INT, 1, size + 1, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecrestoreownershipranges_(Vec *v, F90Array1d *ptr, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
}
