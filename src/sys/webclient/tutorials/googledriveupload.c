
/*
    Run with -google_refresh_token XXX to allow access to your Google Drive or else it will prompt you to enter log in information for Google Drive.
*/

#include <petscsys.h>

int main(int argc, char **argv)
{
  char access_token[512];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscGoogleDriveRefresh(PETSC_COMM_WORLD, NULL, access_token, sizeof(access_token)));
  PetscCall(PetscGoogleDriveUpload(PETSC_COMM_WORLD, access_token, "googledriveupload.c"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: ssl

   test:
     TODO: determine how to run this test without making a google refresh token public

TEST*/
