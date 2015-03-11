
/*
     Shows any outstanding Globus file requests

     You can run PETSc programs with -globus_access_token XXXX where XXX is the access token to access Globus

*/

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           buff[4096];

  PetscInitialize(&argc,&argv,NULL,NULL);
  ierr = PetscGlobusGetTransfers(PETSC_COMM_WORLD,NULL,buff,sizeof(buff));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Transfers are %s\n",buff);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}


