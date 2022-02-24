
/*
     Shows any outstanding Globus file requests

     You can run PETSc programs with -globus_access_token XXXX where XXX is the access token to access Globus

*/

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           buff[4096];

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);if (ierr) return ierr;
  CHKERRQ(PetscGlobusGetTransfers(PETSC_COMM_WORLD,NULL,buff,sizeof(buff)));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Transfers are %s\n",buff));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: ssl

   test:
     TODO: determine how to run this test without going through the browser

TEST*/
