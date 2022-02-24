
/*
     Obtains a refresh token that you can use in the future to access Box from PETSc code

     You can run PETSc programs with -box_refresh_token XXXX where XXX is the refresh token to access your Box

*/

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           access_token[512],refresh_token[512];

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);if (ierr) return ierr;
  CHKERRQ(PetscBoxAuthorize(PETSC_COMM_WORLD,access_token,refresh_token,sizeof(access_token)));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Your one time refresh token is %s\n",refresh_token));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: ssl saws

   test:
     TODO: determine how to run this test without going through the browser

TEST*/
