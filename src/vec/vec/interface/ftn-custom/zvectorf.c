#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define vecsetvalue_         VECSETVALUE
  #define vecsetvaluelocal_    VECSETVALUELOCAL
  #define vecgetarray_         VECGETARRAY
  #define vecgetarrayread_     VECGETARRAYREAD
  #define vecgetarrayaligned_  VECGETARRAYALIGNED
  #define vecrestorearray_     VECRESTOREARRAY
  #define vecrestorearrayread_ VECRESTOREARRAYREAD
  #define vecduplicatevecs_    VECDUPLICATEVECS
  #define vecdestroyvecs_      VECDESTROYVECS
  #define vecmin1_             VECMIN1
  #define vecmin2_             VECMIN2
  #define vecmax1_             VECMAX1
  #define vecmax2_             VECMAX2

#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define vecgetarrayaligned_  vecgetarrayaligned
  #define vecsetvalue_         vecsetvalue
  #define vecsetvaluelocal_    vecsetvaluelocal
  #define vecgetarray_         vecgetarray
  #define vecrestorearray_     vecrestorearray
  #define vecgetarrayaligned_  vecgetarrayaligned
  #define vecgetarrayread_     vecgetarrayread
  #define vecrestorearrayread_ vecrestorearrayread
  #define vecduplicatevecs_    vecduplicatevecs
  #define vecdestroyvecs_      vecdestroyvecs
  #define vecmin1_             vecmin1
  #define vecmin2_             vecmin2
  #define vecmax1_             vecmax1
  #define vecmax2_             vecmax2
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

/*MC
         VecGetArrayAligned - FORTRAN only. Forces alignment of vector
      arrays so that arrays of derived types may be used.

   Synopsis:
   VecGetArrayAligned(PetscErrorCode ierr)

     Not Collective

     Level: advanced

     Notes:
    Allows code such as

.vb
     type  :: Field
        PetscScalar :: p1
        PetscScalar :: p2
      end type Field

      type(Field)       :: lx_v(0:1)

      call VecGetArray(localX, lx_v, lx_i, ierr)
      call InitialGuessLocal(lx_v(lx_i/2), ierr)

      subroutine InitialGuessLocal(a,ierr)
      type(Field)     :: a(*)
.ve

     If you have not called `VecGetArrayAligned()` the code may generate incorrect data
     or crash.

     lx_i needs to be divided by the number of entries in Field (in this case 2)

     You do NOT need `VecGetArrayAligned()` if lx_v and a are arrays of `PetscScalar`

.seealso: `VecGetArray()`, `VecGetArrayF90()`
M*/
static PetscBool  VecGetArrayAligned = PETSC_FALSE;
PETSC_EXTERN void vecgetarrayaligned_(PetscErrorCode *ierr)
{
  VecGetArrayAligned = PETSC_TRUE;
}

PETSC_EXTERN void vecgetarray_(Vec *x, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  PetscScalar *lx;
  PetscInt     m, bs;

  *ierr = VecGetArray(*x, &lx);
  if (*ierr) return;
  *ierr = VecGetLocalSize(*x, &m);
  if (*ierr) return;
  bs = 1;
  if (VecGetArrayAligned) {
    *ierr = VecGetBlockSize(*x, &bs);
    if (*ierr) return;
  }
  *ierr = PetscScalarAddressToFortran((PetscObject)*x, bs, fa, lx, m, ia);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
PETSC_EXTERN void vecrestorearray_(Vec *x, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  PetscInt     m;
  PetscScalar *lx;

  *ierr = VecGetLocalSize(*x, &m);
  if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*x, fa, *ia, m, &lx);
  if (*ierr) return;
  *ierr = VecRestoreArray(*x, &lx);
  if (*ierr) return;
}

PETSC_EXTERN void vecgetarrayread_(Vec *x, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  const PetscScalar *lx;
  PetscInt           m, bs;

  *ierr = VecGetArrayRead(*x, &lx);
  if (*ierr) return;
  *ierr = VecGetLocalSize(*x, &m);
  if (*ierr) return;
  bs = 1;
  if (VecGetArrayAligned) {
    *ierr = VecGetBlockSize(*x, &bs);
    if (*ierr) return;
  }
  *ierr = PetscScalarAddressToFortran((PetscObject)*x, bs, fa, (PetscScalar *)lx, m, ia);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
PETSC_EXTERN void vecrestorearrayread_(Vec *x, PetscScalar *fa, size_t *ia, PetscErrorCode *ierr)
{
  PetscInt           m;
  const PetscScalar *lx;

  *ierr = VecGetLocalSize(*x, &m);
  if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*x, fa, *ia, m, (PetscScalar **)&lx);
  if (*ierr) return;
  *ierr = VecRestoreArrayRead(*x, &lx);
  if (*ierr) return;
}

/*
      vecduplicatevecs() and vecdestroyvecs() are slightly different from C since the
    Fortran provides the array to hold the vector objects,while in C that
    array is allocated by the VecDuplicateVecs()
*/
PETSC_EXTERN void vecduplicatevecs_(Vec *v, PetscInt *m, Vec *newv, PetscErrorCode *ierr)
{
  Vec     *lV;
  PetscInt i;
  *ierr = VecDuplicateVecs(*v, *m, &lV);
  if (*ierr) return;
  for (i = 0; i < *m; i++) newv[i] = lV[i];
  *ierr = PetscFree(lV);
}

PETSC_EXTERN void vecdestroyvecs_(PetscInt *m, Vec *vecs, PetscErrorCode *ierr)
{
  PetscInt i;
  for (i = 0; i < *m; i++) {
    *ierr = VecDestroy(&vecs[i]);
    if (*ierr) return;
  }
}

PETSC_EXTERN void vecmin1_(Vec *x, PetscInt *p, PetscReal *val, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMin(*x, p, val);
}

PETSC_EXTERN void vecmin2_(Vec *x, PetscInt *p, PetscReal *val, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMin(*x, p, val);
}

PETSC_EXTERN void vecmax1_(Vec *x, PetscInt *p, PetscReal *val, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMax(*x, p, val);
}

PETSC_EXTERN void vecmax2_(Vec *x, PetscInt *p, PetscReal *val, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMax(*x, p, val);
}

PETSC_EXTERN void vecgetownershipranges_(Vec *x, PetscInt *range, PetscErrorCode *ierr)
{
  PetscMPIInt     size, mpi_ierr;
  const PetscInt *r;

  mpi_ierr = MPI_Comm_size(PetscObjectComm((PetscObject)*x), &size);
  if (mpi_ierr) {
    *ierr = PETSC_ERR_MPI;
    return;
  }
  *ierr = VecGetOwnershipRanges(*x, &r);
  if (*ierr) return;
  *ierr = PetscArraycpy(range, r, size + 1);
}
