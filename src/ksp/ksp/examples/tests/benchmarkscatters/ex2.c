
static char help[] = "Tests shared memory subcommunicators\n\n";
#include <petscsys.h>

extern PetscErrorCode  PetscCommSplitShared(MPI_Comm,MPI_Comm*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  MPI_Comm       scomm;
  
  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = PetscCommSplitShared(PETSC_COMM_WORLD,&scomm);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&scomm);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
