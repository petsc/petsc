
#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matsetvalue_                   MATSETVALUE
  #define matsetvaluelocal_              MATSETVALUELOCAL
  #define matdiagonalscale_              MATDIAGONALSCALE
  #define matsetpreallocationcoo32_      MATSETPREALLOCATIONCOO32
  #define matsetpreallocationcoolocal32_ MATSETPREALLOCATIONCOOLOCAL32
  #define matsetpreallocationcoo64_      MATSETPREALLOCATIONCOO64
  #define matsetpreallocationcoolocal64_ MATSETPREALLOCATIONCOOLOCAL64
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matsetvalue_                   matsetvalue
  #define matsetvaluelocal_              matsetvaluelocal
  #define matdiagonalscale_              matdiagonalscale
  #define matsetpreallocationcoo32_      matsetpreallocationcoo32
  #define matsetpreallocationcoolocal32_ matsetpreallocationcoolocal32
  #define matsetpreallocationcoo64       matsetpreallocationcoo64
  #define matsetpreallocationcoolocal64_ matsetpreallocationcoolocal64
#endif

PETSC_EXTERN void matsetvalue_(Mat *mat, PetscInt *i, PetscInt *j, PetscScalar *va, InsertMode *mode, PetscErrorCode *ierr)
{
  /* cannot use MatSetValue() here since that uses PetscCall() which has a return in it */
  *ierr = MatSetValues(*mat, 1, i, 1, j, va, *mode);
}

PETSC_EXTERN void matsetvaluelocal_(Mat *mat, PetscInt *i, PetscInt *j, PetscScalar *va, InsertMode *mode, PetscErrorCode *ierr)
{
  /* cannot use MatSetValueLocal() here since that uses PetscCall() which has a return in it */
  *ierr = MatSetValuesLocal(*mat, 1, i, 1, j, va, *mode);
}

PETSC_EXTERN void matsetpreallocationcoo32_(Mat *A, int *ncoo, PetscInt coo_i[], PetscInt coo_j[], int *ierr)
{
  *ierr = MatSetPreallocationCOO(*A, *ncoo, coo_i, coo_j);
}

PETSC_EXTERN void matsetpreallocationcoo64_(Mat *A, PetscInt64 *ncoo, PetscInt coo_i[], PetscInt coo_j[], int *ierr)
{
  *ierr = MatSetPreallocationCOO(*A, *ncoo, coo_i, coo_j);
}

PETSC_EXTERN void matsetpreallocationcoolocal32_(Mat *A, int *ncoo, PetscInt coo_i[], PetscInt coo_j[], int *ierr)
{
  *ierr = MatSetPreallocationCOOLocal(*A, *ncoo, coo_i, coo_j);
}

PETSC_EXTERN void matsetpreallocationcoolocal64_(Mat *A, PetscInt64 *ncoo, PetscInt coo_i[], PetscInt coo_j[], int *ierr)
{
  *ierr = MatSetPreallocationCOOLocal(*A, *ncoo, coo_i, coo_j);
}
