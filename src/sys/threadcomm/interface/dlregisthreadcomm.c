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
  ierr = PetscFunctionListDestroy(&PetscThreadCommList);CHKERRQ(ierr);
  ierr = MPI_Keyval_free(&Petsc_ThreadComm_keyval);CHKERRQ(ierr);
  PetscThreadCommPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_CopyThreadComm"
/*
  This frees the thread communicator attached to MPI_Comm

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Keyval_create()
*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_CopyThreadComm(MPI_Comm comm,PetscMPIInt keyval,void *extra_state,void *attr_in,void *attr_out,int *flag)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm = (PetscThreadComm)attr_in;

  PetscFunctionBegin;
  tcomm->refct++;
  *(void**)attr_out = tcomm;

  *flag = 1;
  ierr  = PetscInfo1(0,"Copying thread communicator data in an MPI_Comm %ld\n",(long)comm);CHKERRQ(ierr);
  if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Petsc_DelThreadComm"
/*
  This frees the thread communicator attached to MPI_Comm

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Keyval_create()
*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelThreadComm(MPI_Comm comm,PetscMPIInt keyval,void *attr,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscThreadCommDestroy((PetscThreadComm*)&attr);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Deleting thread communicator data in an MPI_Comm %ld\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitializePackage"
/*@C
   PetscThreadCommInitializePackage - Initializes ThreadComm package

   Logically collective

   Level: developer

.seealso: PetscThreadCommFinalizePackage()
@*/
PetscErrorCode PetscThreadCommInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscThreadCommPackageInitialized) PetscFunctionReturn(0);

  if (Petsc_ThreadComm_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(Petsc_CopyThreadComm,Petsc_DelThreadComm,&Petsc_ThreadComm_keyval,(void*)0);CHKERRQ(ierr);
  }

  ierr = PetscGetNCores(NULL);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("ThreadCommRunKer",  0, &ThreadComm_RunKernel);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ThreadCommBarrie",  0, &ThreadComm_Barrier);CHKERRQ(ierr);

  PetscThreadCommPackageInitialized = PETSC_TRUE;

  ierr = PetscRegisterFinalize(PetscThreadCommFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
