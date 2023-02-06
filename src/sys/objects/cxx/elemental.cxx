#include <petscsys.h>
#include <petscmatelemental.h>
#include <petsc/private/petscimpl.h>

/*@
   PetscElementalInitializePackage - Initialize Elemental package

   Collective on `MPI_COMM_WORLD`, not `PETSC_COMM_WORLD`

   Level: developer

   Note:
   Can be called outside of `PetscInitialize()` and `PetscFinalize()`.
   If called outside of these functions, it is the user's responsibility
   to make sure that `PETSC_COMM_WORLD` is either unset (default value is `MPI_COMM_NULL`),
   or that it is not `MPI_UNEQUAL` to `MPI_COMM_WORLD`.
   Users who do not have a custom `PETSC_COMM_WORLD` do not have to call this function.

.seealso: `MATELEMENTAL`, `PetscElementalFinalizePackage()`
@*/
PetscErrorCode PetscElementalInitializePackage(void)
{
  if (El::Initialized()) return PETSC_SUCCESS;
  if (PETSC_COMM_WORLD != MPI_COMM_NULL) { /* MPI has been initialized and PETSC_COMM_WORLD has been set */
    PetscMPIInt result;
    PetscCallMPI(MPI_Comm_compare(PETSC_COMM_WORLD, MPI_COMM_WORLD, &result));
    if (result == MPI_UNEQUAL) return PETSC_ERR_MPI; /* cannot use Elemental with PETSC_COMM_WORLD and MPI_COMM_WORLD comparing to MPI_UNEQUAL, call PetscElementalInitializePackage()/PetscElementalFinalizePackage() collectively */
  }
  El::Initialize();            /* called by PetscInitialize_DynamicLibraries(void) or users */
  if (PetscInitializeCalled) { /* true if MPI is initialized by PETSc, false if MPI has been initialized outside and thus PETSC_COMM_WORLD can't be set to something else than MPI_COMM_NULL, see src/sys/objects/pinit.c */
    PetscCall(PetscRegisterFinalize(PetscElementalFinalizePackage));
  }
  return PETSC_SUCCESS;
}

/*@
   PetscElementalInitialized - Determine whether Elemental is initialized

   Not Collective

   Level: developer

   Note:
   Can be called outside of `PetscInitialize()` and `PetscFinalize()`.

.seealso: `MATELEMENTAL`, `PetscElementalInitializePackage()`
@*/
PetscErrorCode PetscElementalInitialized(PetscBool *isInitialized)
{
  if (isInitialized) *isInitialized = (PetscBool)El::Initialized();
  return PETSC_SUCCESS;
}

/*@
   PetscElementalFinalizePackage - Finalize Elemental package

   Collective on `MPI_COMM_WORLD`, not `PETSC_COMM_WORLD`

   Level: developer

   Note:
   Can be called outside of `PetscInitialize()` and `PetscFinalize()`.
   Users who do not call `PetscElementalInitializePackage()` do not have to call this function.

.seealso: `MATELEMENTAL`, `PetscElementalInitializePackage()`
@*/
PetscErrorCode PetscElementalFinalizePackage(void)
{
  if (El::Initialized()) El::Finalize();
  return PETSC_SUCCESS;
}
