
/*
     Obtains a refresh token that you can use in the future to access Google Drive from PETSc code

     Guard the refresh token like a password.

     You can run PETSc programs with -google_refresh_token XXXX where XXX is the refresh token to access your Google Drive

*/

#include <petscsys.h>

int main(int argc,char **argv)
{
  char           access_token[512],refresh_token[512];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  PetscCall(PetscGoogleDriveAuthorize(PETSC_COMM_WORLD,access_token,refresh_token,sizeof(access_token)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Your Refresh token is %s\n",refresh_token));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: ssl

   test:
     TODO: determine how to run this test without going through the browser

TEST*/
