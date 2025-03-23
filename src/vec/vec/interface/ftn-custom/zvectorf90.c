#include <petscvec.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define vecgetarraywrite_     VECGETARRAYWRITE
  #define vecrestorearraywrite_ VECRESTOREARRAYWRITE
  #define vecgetarray_          VECGETARRAY
  #define vecrestorearray_      VECRESTOREARRAY
  #define vecgetarrayread_      VECGETARRAYREAD
  #define vecrestorearrayread_  VECRESTOREARRAYREAD
  #define vecduplicatevecs_     VECDUPLICATEVECS
  #define vecdestroyvecs_       VECDESTROYVECS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define vecgetarraywrite_     vecgetarraywrite
  #define vecrestorearraywrite_ vecrestorearraywrite
  #define vecgetarray_          vecgetarray
  #define vecrestorearray_      vecrestorearray
  #define vecgetarrayread_      vecgetarrayread
  #define vecrestorearrayread_  vecrestorearrayread
  #define vecduplicatevecs_     vecduplicatevecs
  #define vecdestroyvecs_       vecdestroyvecs
#endif

PETSC_EXTERN void vecgetarraywrite_(Vec *x, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     len;
  if (!ptr) {
    *ierr = PetscError(((PetscObject)*x)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "ptr==NULL, maybe #include <petsc/finclude/petscvec.h> is missing?");
    return;
  }
  *ierr = VecGetArrayWrite(*x, &fa);
  if (*ierr) return;
  *ierr = VecGetLocalSize(*x, &len);
  if (*ierr) return;
  *ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, len, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecrestorearraywrite_(Vec *x, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = VecRestoreArrayWrite(*x, &fa);
}

PETSC_EXTERN void vecgetarray_(Vec *x, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     len;
  if (!ptr) {
    *ierr = PetscError(((PetscObject)*x)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "ptr==NULL, maybe #include <petsc/finclude/petscvec.h> is missing?");
    return;
  }
  *ierr = VecGetArray(*x, &fa);
  if (*ierr) return;
  *ierr = VecGetLocalSize(*x, &len);
  if (*ierr) return;
  *ierr = F90Array1dCreate(fa, MPIU_SCALAR, 1, len, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecrestorearray_(Vec *x, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = VecRestoreArray(*x, &fa);
}

PETSC_EXTERN void vecgetarrayread_(Vec *x, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  PetscInt           len;
  if (!ptr) {
    *ierr = PetscError(((PetscObject)*x)->comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_ARG_BADPTR, PETSC_ERROR_INITIAL, "ptr==NULL, maybe #include <petsc/finclude/petscvec.h> is missing?");
    return;
  }
  *ierr = VecGetArrayRead(*x, &fa);
  if (*ierr) return;
  *ierr = VecGetLocalSize(*x, &len);
  if (*ierr) return;
  *ierr = F90Array1dCreate((PetscScalar *)fa, MPIU_SCALAR, 1, len, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecrestorearrayread_(Vec *x, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscScalar *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = VecRestoreArrayRead(*x, &fa);
}

PETSC_EXTERN void vecduplicatevecs_(Vec *v, int *m, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  Vec              *lV;
  PetscFortranAddr *newvint;
  int               i;
  *ierr = VecDuplicateVecs(*v, *m, &lV);
  if (*ierr) return;
  *ierr = PetscMalloc1(*m, &newvint);
  if (*ierr) return;

  for (i = 0; i < *m; i++) newvint[i] = (PetscFortranAddr)lV[i];
  *ierr = PetscFree(lV);
  if (*ierr) return;
  *ierr = F90Array1dCreate(newvint, MPIU_FORTRANADDR, 1, *m, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecdestroyvecs_(int *m, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  Vec *vecs;
  int  i;

  *ierr = F90Array1dAccess(ptr, MPIU_FORTRANADDR, (void **)&vecs PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  for (i = 0; i < *m; i++) {
    PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(&vecs[i]);
    *ierr = VecDestroy(&vecs[i]);
    if (*ierr) return;
    PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(&vecs[i]);
  }
  *ierr = F90Array1dDestroy(ptr, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = PetscFree(vecs);
}
