#include <petsc-private/threadcommimpl.h>

static PetscBool PetscThreadCommPackageInitialized = PETSC_FALSE;

extern PetscErrorCode PetscThreadCommDetach(MPI_Comm);
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommFinalizePackage"
/*@C
   PetscThreadCommFinalizePackage - Finalize PetscThreadComm package, called from PetscFinalize()

   Logically collective

   Level: developer

.seealso: PetscThreadCommInitializePackage()
@*/
PetscErrorCode PetscThreadCommFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscThreadCommRegisterDestroy();CHKERRQ(ierr);

  ierr = PetscThreadCommDetach(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscThreadCommDetach(PETSC_COMM_SELF);CHKERRQ(ierr);

  ierr = MPI_Keyval_free(&Petsc_ThreadComm_keyval);CHKERRQ(ierr);
  PetscThreadCommPackageInitialized = PETSC_FALSE;
  PetscThreadCommList = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitializePackage"
/*@C
   PetscThreadCommInitializePackage - Initializes ThreadComm package

   Logically collective

   Input Parameter:
.  path - The dynamic library path, or PETSC_NULL

   Level: developer

.seealso: PetscThreadCommFinalizePackage()
@*/
PetscErrorCode PetscThreadCommInitializePackage(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscThreadCommPackageInitialized) PetscFunctionReturn(0);
  ierr = PetscLogEventRegister("ThreadCommRunKernel",  0, &ThreadComm_RunKernel);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ThreadCommBarrier",    0, &ThreadComm_Barrier);CHKERRQ(ierr);
  ierr = PetscThreadCommInitialize();CHKERRQ(ierr);
  PetscThreadCommPackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscThreadCommFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
