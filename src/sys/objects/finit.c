#include <petsc/private/petscimpl.h> /*I  "petscsys.h"   I*/

#if defined(PETSC_USE_FORTRAN_BINDINGS)
  #if defined(PETSC_HAVE_FORTRAN_CAPS)
    #define petscinitializefortran_     PETSCINITIALIZEFORTRAN
    #define petscsetmoduleblock_        PETSCSETMODULEBLOCK
    #define petscsetmoduleblockmpi_     PETSCSETMODULEBLOCKMPI
    #define petscsetmoduleblocknumeric_ PETSCSETMODULEBLOCKNUMERIC
    #define petscsetcomm_               PETSCSETCOMM
  #elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
    #define petscinitializefortran_     petscinitializefortran
    #define petscsetmoduleblock_        petscsetmoduleblock
    #define petscsetmoduleblockmpi_     petscsetmoduleblockmpi
    #define petscsetmoduleblocknumeric_ petscsetmoduleblocknumeric
    #define petscsetcomm_               petscsetcomm
  #endif

PETSC_EXTERN void petscsetmoduleblock_(void);
PETSC_EXTERN void petscsetmoduleblockmpi_(MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *);
PETSC_EXTERN void petscsetmoduleblocknumeric_(PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN void petscsetcomm_(MPI_Fint *, MPI_Fint *);
#endif

/*@C
  PetscInitializeFortran - Routine that should be called soon AFTER
  the call to `PetscInitialize()` if one is using a C main program
  that calls Fortran routines that in turn call PETSc routines.

  Collective on `PETSC_COMM_WORLD`

  Level: beginner

  Notes:
  `PetscInitializeFortran()` initializes some of the default viewers,
  communicators, etc. for use in the Fortran if a user's main program is
  written in C.  `PetscInitializeFortran()` is NOT needed if a user's main
  program is written in Fortran; in this case, just calling
  `PetscInitialize()` in the main (Fortran) program is sufficient.

  This function exists and can be called even if PETSc has been configured
  with `--with-fortran-bindings=0` or `--with-fc=0`. It just does nothing
  in that case.

.seealso: `PetscInitialize()`
@*/
PetscErrorCode PetscInitializeFortran(void)
{
#if defined(PETSC_USE_FORTRAN_BINDINGS)
  MPI_Fint c1 = 0, c2 = 0;

  if (PETSC_COMM_WORLD) c1 = MPI_Comm_c2f(PETSC_COMM_WORLD);
  c2 = MPI_Comm_c2f(PETSC_COMM_SELF);
  petscsetmoduleblock_();
  petscsetcomm_(&c1, &c2);

  {
    MPI_Fint freal, fscalar, fsum, fint;
    freal   = MPI_Type_c2f(MPIU_REAL);
    fscalar = MPI_Type_c2f(MPIU_SCALAR);
    fsum    = MPI_Op_c2f(MPIU_SUM);
    fint    = MPI_Type_c2f(MPIU_INT);
    petscsetmoduleblockmpi_(&freal, &fscalar, &fsum, &fint);
  }

  {
    PetscReal pi      = PETSC_PI;
    PetscReal maxreal = PETSC_MAX_REAL;
    PetscReal minreal = PETSC_MIN_REAL;
    PetscReal eps     = PETSC_MACHINE_EPSILON;
    PetscReal seps    = PETSC_SQRT_MACHINE_EPSILON;
    PetscReal small   = PETSC_SMALL;
    PetscReal pinf    = PETSC_INFINITY;
    PetscReal pninf   = PETSC_NINFINITY;
    petscsetmoduleblocknumeric_(&pi, &maxreal, &minreal, &eps, &seps, &small, &pinf, &pninf);
  }
#endif
  return PETSC_SUCCESS;
}
