
/*
     Shows any outstanding Globus file requests

     You can run PETSc programs with -globus_access_token XXXX where XXX is the access token to access Globus

*/

#include <petscsys.h>

int main(int argc,char **argv)
{
  char           buff[4096];

  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  PetscCall(PetscGlobusGetTransfers(PETSC_COMM_WORLD,NULL,buff,sizeof(buff)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Transfers are %s\n",buff));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: ssl

   test:
     TODO: determine how to run this test without going through the browser

TEST*/
