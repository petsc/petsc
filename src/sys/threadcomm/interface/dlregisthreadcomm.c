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
  MPI_Comm        icomm;
  void            *ptr;
  PetscMPIInt     flg;
  PetscFunctionBegin;
  ierr = PetscThreadCommRegisterDestroy();CHKERRQ(ierr);

  /* Get the inner communicator */
  ierr  = MPI_Attr_get(PETSC_COMM_WORLD,Petsc_InnerComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (flg) {
    /*  Use PetscMemcpy() because casting from pointer to integer of different size is not allowed with some compilers  */
    ierr = PetscMemcpy(&icomm,&ptr,sizeof(MPI_Comm));CHKERRQ(ierr);
    /* Delete the thread communicator */
    ierr = MPI_Attr_delete(icomm,Petsc_ThreadComm_keyval);CHKERRQ(ierr);
  }

  /* Free the thread communicator key */
  ierr = MPI_Keyval_free(&Petsc_ThreadComm_keyval);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&icomm);CHKERRQ(ierr);
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
  ierr = PetscThreadCommInitialize();CHKERRQ(ierr);
  PetscThreadCommPackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscThreadCommFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
