#include <../src/sys/threadcomm/impls/nothread/nothreadimpl.h>

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_NoThread"
PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm tcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(tcomm->nworkThreads != 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot have more than 1 thread for the nonthread communicator,threads requested = %D",tcomm->nworkThreads);
  ierr = PetscStrcpy(tcomm->type,NOTHREAD);CHKERRQ(ierr);
  tcomm->ops->runkernel = PetscThreadCommRunKernel_NoThread;
  PetscFunctionReturn(0);
}
EXTERN_C_END
   
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_NoThread"
PetscErrorCode PetscThreadCommRunKernel_NoThread(MPI_Comm comm,PetscThreadCommJobCtx job)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscRunKernel(0,job->nargs,job);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
