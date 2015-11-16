
static char help[] = "Tests shared memory subcommunicators\n\n";
#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  PetscCommShared scomm;
  MPI_Comm        comm;
  PetscMPIInt     lrank;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&comm,NULL);CHKERRQ(ierr);
  ierr = PetscCommSharedGet(comm,&scomm);CHKERRQ(ierr);
  ierr = PetscCommSharedGet(comm,&scomm);CHKERRQ(ierr);

  ierr = PetscCommSharedGlobalToLocal(scomm,1,&lrank);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Global rank %d shared memory comm rank %d\n",1,lrank);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
