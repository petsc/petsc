
/*
     Obtains a refresh token that you can use in the future to access Box from PETSc code

     Note: this does not work, see PetscBoxAuthorize()
     Guard the refresh token like a password.

     You can run PETSc programs with -box_refresh_token XXXX where XXX is the refresh token to access your Box

*/

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           access_token[512],refresh_token[512];

  PetscInitialize(&argc,&argv,NULL,NULL);
  ierr = PetscBoxAuthorize(PETSC_COMM_WORLD,access_token,refresh_token,sizeof(access_token));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Your one time refresh token is %s\n",refresh_token);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}


