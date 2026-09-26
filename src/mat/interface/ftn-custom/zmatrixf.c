#include <petsc/private/ftnimpl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define matdestroymatrices_      MATDESTROYMATRICES
  #define matdestroysubmatrices_   MATDESTROYSUBMATRICES
  #define matcreatesubmatrices_    MATCREATESUBMATRICES
  #define matcreatesubmatricesmpi_ MATCREATESUBMATRICESMPI
  #define matnullspacesetfunction_ MATNULLSPACESETFUNCTION
  #define matfindnonzerorows_      MATFINDNONZEROROWS
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define matdestroymatrices_      matdestroymatrices
  #define matdestroysubmatrices_   matdestroysubmatrices
  #define matcreatesubmatrices_    matcreatesubmatrices
  #define matcreatesubmatricesmpi_ matcreatesubmatricesmpi
  #define matnullspacesetfunction_ matnullspacesetfunction
  #define matfindnonzerorows_      matfindnonzerorows
#endif

static PetscErrorCode ournullfunction(MatNullSpace sp, Vec x, PetscCtx ctx)
{
  PetscCallFortranVoidFunction((*(void (*)(MatNullSpace *, Vec *, void *, PetscErrorCode *))(((PetscObject)sp)->fortran_func_pointers[0]))(&sp, &x, ctx, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void matnullspacesetfunction_(MatNullSpace *sp, PetscErrorCode (*rem)(MatNullSpace, Vec, void *), PetscCtx ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*sp, 1);
  ((PetscObject)*sp)->fortran_func_pointers[0] = (PetscFortranCallbackFn *)rem;

  *ierr = MatNullSpaceSetFunction(*sp, ournullfunction, ctx);
}

PETSC_EXTERN void matcreatesubmatrices_(Mat *mat, PetscInt *n, IS *irow, IS *icol, MatReuse *scall, F90Array1d *submat, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  Mat *lsmat;

  if (*scall == MAT_INITIAL_MATRIX) {
    *ierr = MatCreateSubMatrices(*mat, *n, irow, icol, *scall, &lsmat);
    *ierr = F90Array1dCreate(lsmat, MPIU_FORTRANADDR, 1, *n + 1, submat PETSC_F90_2PTR_PARAM(ptrd));
  } else {
    *ierr = F90Array1dAccess(submat, MPIU_FORTRANADDR, (void **)&lsmat PETSC_F90_2PTR_PARAM(ptrd));
    *ierr = MatCreateSubMatrices(*mat, *n, irow, icol, *scall, &lsmat);
  }
}

PETSC_EXTERN void matcreatesubmatricesmpi_(Mat *mat, PetscInt *n, IS *irow, IS *icol, MatReuse *scall, F90Array1d *submat, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  Mat *lsmat;

  if (*scall == MAT_INITIAL_MATRIX) {
    *ierr = MatCreateSubMatricesMPI(*mat, *n, irow, icol, *scall, &lsmat);
    if (*ierr) return;
    *ierr = F90Array1dCreate(lsmat, MPIU_FORTRANADDR, 1, *n + 1, submat PETSC_F90_2PTR_PARAM(ptrd));
  } else {
    *ierr = F90Array1dAccess(submat, MPIU_FORTRANADDR, (void **)&lsmat PETSC_F90_2PTR_PARAM(ptrd));
    if (*ierr) return;
    *ierr = MatCreateSubMatricesMPI(*mat, *n, irow, icol, *scall, &lsmat);
  }
}

PETSC_EXTERN void matdestroymatrices_(PetscInt *n, F90Array1d *mat, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  Mat *lsmat;

  *ierr = F90Array1dAccess(mat, MPIU_FORTRANADDR, (void **)&lsmat PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  for (PetscInt i = 0; i < *n; i++) {
    PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(&lsmat[i]);
    *ierr = MatDestroy(&lsmat[i]);
    if (*ierr) return;
  }
  *ierr = F90Array1dDestroy(mat, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = PetscFree(lsmat);
}

PETSC_EXTERN void matdestroysubmatrices_(PetscInt *n, F90Array1d *mat, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  Mat *lsmat;

  if (*n == 0) return;
  *ierr = F90Array1dAccess(mat, MPIU_FORTRANADDR, (void **)&lsmat PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = MatDestroySubMatrices(*n, &lsmat);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(mat, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = PetscFree(lsmat);
}
