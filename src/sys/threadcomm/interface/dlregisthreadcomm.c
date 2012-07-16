#include <petsc-private/threadcommimpl.h>

static PetscBool PetscThreadCommPackageInitialized = PETSC_FALSE;

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
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = PetscThreadCommRegisterDestroy();CHKERRQ(ierr);

#if !defined(PETSC_HAVE_MPIUNI)
  comm = PETSC_COMM_WORLD;      /* Release reference from PetscThreadCommInitialize */
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
#endif

  comm = PETSC_COMM_SELF;       /* Release reference from PetscThreadCommInitialize */
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = MPI_Keyval_free(&Petsc_ThreadComm_keyval);CHKERRQ(ierr);
  PetscThreadCommPackageInitialized = PETSC_FALSE;
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
  if(PetscThreadCommPackageInitialized) PetscFunctionReturn(0);
  ierr = PetscLogEventRegister("ThreadCommRunKernel",  0, &ThreadComm_RunKernel);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ThreadCommBarrier",    0, &ThreadComm_Barrier);CHKERRQ(ierr);
  ierr = PetscThreadCommInitialize();CHKERRQ(ierr);
  PetscThreadCommPackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscThreadCommFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
