
static char help[] = "Demonstrates PETSc error handlers.\n";

#include <petscsys.h>

int CreateError(int n)
{
  PetscErrorCode ierr;
  if (!n) SETERRQ(PETSC_COMM_SELF,1,"Error Created");
  ierr = CreateError(n-1);CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Demonstrates PETSc Error Handlers\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"The error is a contrived error to test error handling\n");CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  ierr = CreateError(5);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

 # Testing errors so only look for errors
   test:
     filter: Error: egrep "(PETSC ERROR)" | egrep "(main|CreateError|Error Created)" | cut -f1,2,3,4,5,6 -d" "


TEST*/
