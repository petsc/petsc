#include <petsc/private/ftnimpl.h>
#include <petsc/private/matimpl.h>

/* Declare these pointer types instead of void* for clarity, but do not include petscts.h so that this code does have an actual reverse dependency. */
typedef struct _p_TS   *TS;
typedef struct _p_SNES *SNES;

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matfdcoloringsetfunctionts_           MATFDCOLORINGSETFUNCTIONTS
  #define matfdcoloringsetfunction_             MATFDCOLORINGSETFUNCTION
  #define matfdcoloringgetperturbedcolumns_     MATFDCOLORINGGETPERTURBEDCOLUMNS
  #define matfdcoloringrestoreperturbedcolumns_ MATFDCOLORINGRESTOREPERTURBEDCOLUMNS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matfdcoloringsetfunctionts_           matfdcoloringsetfunctionts
  #define matfdcoloringsetfunction_             matfdcoloringsetfunction
  #define matfdcoloringgetperturbedcolumns_     matfdcoloringgetperturbedcolumns
  #define matfdcoloringrestoreperturbedcolumns_ matfdcoloringrestoreperturbedcolumns
#endif

PETSC_EXTERN void matfdcoloringgetperturbedcolumns_(MatFDColoring *x, PetscInt *len, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *__ierr = MatFDColoringGetPerturbedColumns(*x, len, &fa);
  if (*__ierr) return;
  *__ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, *len, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void matfdcoloringrestoreperturbedcolumns_(MatFDColoring *x, PetscInt *len, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *__ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
}

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourmatfdcoloringfunctionts(TS ts, PetscReal t, Vec x, Vec y, MatFDColoring fd)
{
  PetscErrorCode ierr = PETSC_SUCCESS;
  (*(void (*)(TS *, PetscReal *, Vec *, Vec *, void *, PetscErrorCode *))fd->ftn_func_pointer)(&ts, &t, &x, &y, fd->ftn_func_cntx, &ierr);
  return ierr;
}

static PetscErrorCode ourmatfdcoloringfunctionsnes(SNES snes, Vec x, Vec y, MatFDColoring fd)
{
  PetscErrorCode ierr = PETSC_SUCCESS;
  (*(void (*)(SNES *, Vec *, Vec *, void *, PetscErrorCode *))fd->ftn_func_pointer)(&snes, &x, &y, fd->ftn_func_cntx, &ierr);
  return ierr;
}

/*
        MatFDColoringSetFunction sticks the Fortran function and its context into the MatFDColoring structure and passes the MatFDColoring object
    in as the function context. ourmafdcoloringfunctionsnes() and ourmatfdcoloringfunctionts()  then access the function and its context from the
    MatFDColoring that is passed in. This is the same way that fortran_func_pointers is used in PETSc objects.

   NOTE: FORTRAN USER CANNOT PUT IN A NEW J OR B currently.
*/

PETSC_EXTERN void matfdcoloringsetfunctionts_(MatFDColoring *fd, void (*f)(TS *, double *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  (*fd)->ftn_func_pointer = (PetscFortranCallbackFn *)f;
  (*fd)->ftn_func_cntx    = ctx;

  *ierr = MatFDColoringSetFunction(*fd, (MatFDColoringFn *)(PetscVoidFn *)ourmatfdcoloringfunctionts, *fd);
}

PETSC_EXTERN void matfdcoloringsetfunction_(MatFDColoring *fd, void (*f)(SNES *, Vec *, Vec *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  (*fd)->ftn_func_pointer = (PetscFortranCallbackFn *)f;
  (*fd)->ftn_func_cntx    = ctx;

  *ierr = MatFDColoringSetFunction(*fd, (MatFDColoringFn *)ourmatfdcoloringfunctionsnes, *fd);
}
